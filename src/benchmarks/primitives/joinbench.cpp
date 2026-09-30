// run_joinbench: hash join probe variants on TPC-H-shaped joins.
//
// Every variant runs the full probe pipeline per 1024-row vector -- hash the
// probe keys, find build matches, check key equality, gather one payload --
// and must produce the same (probe row, payload) pairs as T0 (count and an
// order-independent checksum).
//
//   T0     today: runtime::Hashmap (tagged directory + chained entries),
//          Hashjoin::joinAllParallel's loop, keys_equal, gather
//   J1     run dedup: probe keys often come in runs of equal keys (lineitem is
//          sorted by l_orderkey); hash + look up once per run, expand the
//          match to the run's rows
//   J2     two-pass (group prefetch): prefetch all directory slots of the
//          vector, then tag-filter + prefetch entries, then compare
//   J2+J1a J2, plus run dedup for vectors where >= 25% of keys repeat their
//          predecessor (gate costs one compare pass)
//   J3     bucketized table: 16 build keys per 64-byte bucket, one SIMD
//          compare per probe (k == ht[pos]); row index in a parallel array;
//          linear probing over buckets, no chains
//   J3P    J3 with a prefetch pass over the vector's buckets
//   J1+J3  run dedup on top of J3
//   J4     Hashjoin::joinAllSIMD's AVX-512 gather loop (x86 AVX-512 only)
//
// Shapes (sf1, int32 keys, MurMurHash with the primitives' seed):
//   q3      Q3 J2: build 10% of 1.5M orders, probe lineitem-like sorted runs (~4/key)
//   q5      Q5 J4: build 15% of 1.5M orders, sorted runs
//   q18     Q18 J1/J3: build ~57 orders, probe lineitem sorted runs (almost no hits)
//   q9      Q9 J4: build 5.4% of 800k partsupp keys, probe random keys
//   q9j5    Q9 J5: build 5.4% of lineitem rows keyed by order (duplicate
//           build keys, chains), probe all orders (unique, sorted)
//   fk_all  full FK join: build all 1.5M orders, probe sorted runs (100% hits)
//   dim     build 10k suppliers, probe random keys
//
// Usage: run_joinbench [-s scale] [-r reps] [-q shape,...]   (CSV on stdout)
#include "common/runtime/Hash.hpp"
#include "common/runtime/Hashmap.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using runtime::Hashmap;
using hash_t = Hashmap::hash_t;
constexpr size_t kVec = 1024;
constexpr uint64_t kSeed = 29875498475984ull; // Primitives.hpp seed, 64-bit hashes
constexpr int32_t kEmpty = INT32_MIN;         // J3 empty slot (never a key)

struct Entry {
   Hashmap::EntryHeader h{nullptr, 0};
   int32_t key = 0;
   int64_t payload = 0;
};

inline hash_t hashKey(int32_t k) { return runtime::MurMurHash().hashKey((uint64_t)(int64_t)k, kSeed); }
inline uint64_t mix(uint64_t pos, int64_t payload) {
   uint64_t x = pos * 0x9E3779B97F4A7C15ull ^ (uint64_t)payload;
   x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
   return x;
}
struct Result {
   uint64_t count = 0, checksum = 0;
   bool operator==(const Result& o) const { return count == o.count && checksum == o.checksum; }
};

//--- workload ------------------------------------------------------------------
struct Workload {
   std::vector<int32_t> buildKeys, probeKeys;
   std::vector<int64_t> buildPayload;
};

Workload makeWorkload(const std::string& shape, double scale, std::mt19937_64& rng) {
   Workload w;
   auto pick = [&](int32_t domain, double frac) {
      for (int32_t k = 0; k < domain; ++k)
         if (frac >= 1.0 || std::uniform_real_distribution<double>(0, 1)(rng) < frac) w.buildKeys.push_back(k);
      std::shuffle(w.buildKeys.begin(), w.buildKeys.end(), rng);
   };
   auto sortedRuns = [&](int32_t orders) { // lineitem: 1..7 rows per order, in key order
      for (int32_t o = 0; o < orders; ++o)
         for (int r = 0, n = 1 + int(rng() % 7); r < n; ++r) w.probeKeys.push_back(o);
   };
   auto randomProbe = [&](int32_t domain, size_t n) {
      w.probeKeys.resize(n);
      for (auto& k : w.probeKeys) k = int32_t(rng() % domain);
   };
   const int32_t orders = int32_t(1500000 * scale);
   if (shape == "q3") pick(orders, 0.10), sortedRuns(orders);
   else if (shape == "q5") pick(orders, 0.15), sortedRuns(orders);
   else if (shape == "fk_all") pick(orders, 1.0), sortedRuns(orders);
   else if (shape == "q18") { // build: the ~57 orders past the sum(l_quantity) filter
      for (int i = 0; i < 57; ++i) w.buildKeys.push_back(int32_t(rng() % orders));
      std::sort(w.buildKeys.begin(), w.buildKeys.end());
      w.buildKeys.erase(std::unique(w.buildKeys.begin(), w.buildKeys.end()), w.buildKeys.end());
      sortedRuns(orders);
   } else if (shape == "q9j5") { // build: lineitem rows (5.4%) keyed by l_orderkey (duplicates); probe: orders
      for (int32_t o = 0; o < orders; ++o)
         for (int r = 0, n = 1 + int(rng() % 7); r < n; ++r)
            if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.054) w.buildKeys.push_back(o);
      std::shuffle(w.buildKeys.begin(), w.buildKeys.end(), rng);
      for (int32_t o = 0; o < orders; ++o) w.probeKeys.push_back(o);
   }
   else if (shape == "q9") pick(int32_t(800000 * scale), 0.054), randomProbe(int32_t(800000 * scale), size_t(6000000 * scale));
   else if (shape == "dim") pick(int32_t(10000 * std::max(scale, 0.01)), 1.0), randomProbe(int32_t(10000 * std::max(scale, 0.01)), size_t(6000000 * scale));
   else { std::fprintf(stderr, "unknown shape %s\n", shape.c_str()); std::exit(1); }
   w.buildPayload.resize(w.buildKeys.size());
   for (auto& p : w.buildPayload) p = int64_t(rng());
   return w;
}

//--- today's table: runtime::Hashmap with chained entries -------------------------
struct ChainTable {
   Hashmap ht;
   std::vector<Entry> entries;
   explicit ChainTable(const Workload& w) : entries(w.buildKeys.size()) {
      for (size_t i = 0; i < entries.size(); ++i) {
         entries[i].key = w.buildKeys[i];
         entries[i].payload = w.buildPayload[i];
         entries[i].h.hash = hashKey(w.buildKeys[i]);
         entries[i].h.next = nullptr;
      }
      ht.setSize(entries.size());
      ht.insertAll_tagged<false>(&entries[0].h, entries.size(), sizeof(Entry));
   }
};

/// per-vector buffers shared by the variants
struct Bufs {
   hash_t hashes[kVec];
   Hashmap::EntryHeader* bm[kVec * 8];
   uint32_t pm[kVec * 8];
   std::vector<std::pair<uint32_t, Hashmap::EntryHeader*>> followups;
   int32_t runKeys[kVec];
   uint32_t runStart[kVec + 1];
   const Hashmap::EntryHeader* cand[kVec];
   uint32_t candIdx[kVec];
};

/// key equality + gather + checksum for (entry, probe index) pairs
inline void finish(const int32_t* keys, size_t base, Hashmap::EntryHeader** bm, const uint32_t* pm,
                   size_t found, Result& res) {
   for (size_t j = 0; j < found; ++j) {
      const Entry* e = reinterpret_cast<const Entry*>(bm[j]);
      if (e->key == keys[pm[j]]) {
         res.count += 1;
         res.checksum += mix(base + pm[j], e->payload);
      }
   }
}

/// joinAllParallel's lookup: first entries, then chains via the followup queue
inline size_t chainLookup(Hashmap& ht, const hash_t* hashes, size_t n, Bufs& b) {
   size_t found = 0;
   b.followups.clear();
   for (size_t i = 0; i < n; ++i) {
      auto entry = ht.find_chain_tagged(hashes[i]);
      if (entry != ht.end()) {
         if (entry->hash == hashes[i]) b.bm[found] = entry, b.pm[found++] = uint32_t(i);
         if (entry->next != ht.end()) b.followups.push_back({uint32_t(i), entry->next});
      }
   }
   for (size_t f = 0; f < b.followups.size(); ++f) {
      auto [i, entry] = b.followups[f];
      if (entry->hash == hashes[i]) b.bm[found] = entry, b.pm[found++] = i;
      if (entry->next != ht.end()) b.followups.push_back({i, entry->next});
   }
   return found;
}

Result probeT0(ChainTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      for (size_t i = 0; i < n; ++i) b.hashes[i] = hashKey(keys[base + i]);
      const size_t found = chainLookup(t.ht, b.hashes, n, b);
      finish(keys + base, base, b.bm, b.pm, found, res);
   }
   return res;
}

/// J1: runs of equal probe keys -> one hash + lookup per run
Result probeJ1(ChainTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      size_t runs = 0;
      for (size_t i = 0; i < n; ++i)
         if (i == 0 || keys[base + i] != keys[base + i - 1]) b.runKeys[runs] = keys[base + i], b.runStart[runs++] = uint32_t(i);
      b.runStart[runs] = uint32_t(n);
      for (size_t r = 0; r < runs; ++r) b.hashes[r] = hashKey(b.runKeys[r]);
      const size_t found = chainLookup(t.ht, b.hashes, runs, b);
      for (size_t j = 0; j < found; ++j) {
         const Entry* e = reinterpret_cast<const Entry*>(b.bm[j]);
         const uint32_t r = b.pm[j];
         if (e->key != b.runKeys[r]) continue;
         for (uint32_t i = b.runStart[r]; i < b.runStart[r + 1]; ++i) {
            res.count += 1;
            res.checksum += mix(base + i, e->payload);
         }
      }
   }
   return res;
}

/// J2's lookup: group prefetching over n hashes. Phase A loads every
/// directory slot (prefetched) with no entry access; phase B touches only the
/// tag-filter candidates (prefetched in phase A), then the chains.
inline size_t twoPassLookup(ChainTable& t, const hash_t* hashes, size_t n, Bufs& b) {
   auto* dir = reinterpret_cast<const uint64_t*>(t.ht.entries);
   const uint64_t mask = t.ht.mask, maskPtr = t.ht.maskPointer;
   for (size_t i = 0; i < n; ++i) __builtin_prefetch(&dir[hashes[i] & mask]);
   size_t nc = 0;
   for (size_t i = 0; i < n; ++i) {
      const uint64_t c = dir[hashes[i] & mask];
      const uint64_t tag = uint64_t(1) << ((hashes[i] >> 60) + 48);
      if (c & tag) {
         const auto* e = reinterpret_cast<const Hashmap::EntryHeader*>(c & maskPtr);
         __builtin_prefetch(e);
         b.cand[nc] = e, b.candIdx[nc++] = uint32_t(i);
      }
   }
   size_t found = 0;
   b.followups.clear();
   for (size_t j = 0; j < nc; ++j) {
      auto* e = const_cast<Hashmap::EntryHeader*>(b.cand[j]);
      const uint32_t i = b.candIdx[j];
      if (e->hash == hashes[i]) b.bm[found] = e, b.pm[found++] = i;
      if (e->next) b.followups.push_back({i, e->next});
   }
   for (size_t f = 0; f < b.followups.size(); ++f) {
      auto [i, e] = b.followups[f];
      if (e->hash == hashes[i]) b.bm[found] = e, b.pm[found++] = i;
      if (e->next) b.followups.push_back({i, e->next});
   }
   return found;
}

/// J2: group prefetching over the vector
Result probeJ2(ChainTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      for (size_t i = 0; i < n; ++i) b.hashes[i] = hashKey(keys[base + i]);
      const size_t found = twoPassLookup(t, b.hashes, n, b);
      finish(keys + base, base, b.bm, b.pm, found, res);
   }
   return res;
}

/// J2+J1a: J2's lookup, with run dedup only for vectors where >= 25% of the
/// keys repeat their predecessor (sorted probe streams); otherwise plain J2
Result probeJ2J1a(ChainTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      const int32_t* k = keys + base;
      size_t runs = 1;
      for (size_t i = 1; i < n; ++i) runs += k[i] != k[i - 1];
      if (runs * 4 > n * 3) { // few repeats: plain J2
         for (size_t i = 0; i < n; ++i) b.hashes[i] = hashKey(k[i]);
         finish(k, base, b.bm, b.pm, twoPassLookup(t, b.hashes, n, b), res);
         continue;
      }
      runs = 0;
      for (size_t i = 0; i < n; ++i)
         if (i == 0 || k[i] != k[i - 1]) b.runKeys[runs] = k[i], b.runStart[runs++] = uint32_t(i);
      b.runStart[runs] = uint32_t(n);
      for (size_t r = 0; r < runs; ++r) b.hashes[r] = hashKey(b.runKeys[r]);
      const size_t found = twoPassLookup(t, b.hashes, runs, b);
      for (size_t j = 0; j < found; ++j) {
         const Entry* e = reinterpret_cast<const Entry*>(b.bm[j]);
         const uint32_t r = b.pm[j];
         if (e->key != b.runKeys[r]) continue;
         for (uint32_t i = b.runStart[r]; i < b.runStart[r + 1]; ++i) {
            res.count += 1;
            res.checksum += mix(base + i, e->payload);
         }
      }
   }
   return res;
}

#if defined(__x86_64__) && defined(__AVX512F__) && !(defined(HASH_SIZE) && HASH_SIZE == 32)
/// J4: Hashjoin::joinAllSIMD's 8-wide loop (64-bit hash path)
Result probeJ4(ChainTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   Hashmap& ht = t.ht;
   std::vector<uint32_t> fuIds(kVec * 8);
   std::vector<Hashmap::EntryHeader*> fuEntries(kVec * 8);
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      for (size_t i = 0; i < n; ++i) b.hashes[i] = hashKey(keys[base + i]);
      size_t found = 0, fw = 0;
      const size_t rest = n % 8;
      auto ids = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
      for (size_t i = 0, end = n - rest; i < end; i += 8) {
         Vec8u hashes(b.hashes + i);
         Vec8uM entries = ht.find_chain_tagged(hashes);
         Vec8u entryHashes = _mm512_mask_i64gather_epi64(
             entries.vec, entries.mask, entries.vec + Vec8u(offsetof(Hashmap::EntryHeader, hash)), nullptr, 1);
         __mmask8 hashesEq = _mm512_mask_cmpeq_epi64_mask(entries.mask, entryHashes, hashes);
         _mm512_mask_compressstoreu_epi64(b.bm + found, hashesEq, entries.vec);
         _mm512_mask_compressstoreu_epi32(b.pm + found, hashesEq, ids);
         found += __builtin_popcount(hashesEq);
         Vec8u nextPtrs = _mm512_mask_i64gather_epi64(entries.vec, entries.mask, entries.vec, nullptr, 1);
         __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(entries.mask, nextPtrs, Vec8u(uint64_t(0)));
         if (hasNext) {
            _mm512_mask_compressstoreu_epi64(fuEntries.data() + fw, hasNext, nextPtrs);
            _mm512_mask_compressstoreu_epi32(fuIds.data() + fw, hasNext, ids);
            fw += __builtin_popcount(hasNext);
         }
         ids = _mm512_add_epi32(ids, _mm512_set1_epi32(8));
      }
      for (size_t i = n - rest; i < n; ++i) {
         auto entry = ht.find_chain_tagged(b.hashes[i]);
         if (entry != ht.end()) {
            if (entry->hash == b.hashes[i]) b.bm[found] = entry, b.pm[found++] = uint32_t(i);
            if (entry->next != ht.end()) fuIds[fw] = uint32_t(i), fuEntries[fw++] = entry->next;
         }
      }
      for (size_t f = 0; f < fw; ++f) {
         const uint32_t i = fuIds[f];
         auto entry = fuEntries[f];
         if (entry->hash == b.hashes[i]) b.bm[found] = entry, b.pm[found++] = i;
         if (entry->next != ht.end()) {
            if (fw == fuIds.size()) fuIds.resize(fw * 2), fuEntries.resize(fw * 2);
            fuIds[fw] = i, fuEntries[fw++] = entry->next;
         }
      }
      finish(keys + base, base, b.bm, b.pm, found, res);
   }
   return res;
}
#define HAVE_J4 1
#endif

//--- J3: bucketized table, 16 keys per 64-byte bucket ------------------------------
struct BucketTable {
   uint64_t mask = 0;
   int32_t* keys = nullptr;    // 16 per bucket, 64-byte aligned
   uint32_t* rows = nullptr;   // parallel: build row of each slot
   std::vector<int64_t> payload;
   explicit BucketTable(const Workload& w) : payload(w.buildPayload) {
      size_t nb = 1;
      while (nb * 16 * 0.7 < w.buildKeys.size()) nb <<= 1;
      mask = nb - 1;
      keys = static_cast<int32_t*>(std::aligned_alloc(64, nb * 64));
      rows = static_cast<uint32_t*>(std::aligned_alloc(64, nb * 64));
      std::fill(keys, keys + nb * 16, kEmpty);
      for (size_t r = 0; r < w.buildKeys.size(); ++r) {
         for (uint64_t bkt = hashKey(w.buildKeys[r]) & mask;; bkt = (bkt + 1) & mask) {
            int32_t* k = keys + bkt * 16;
            int s = 0;
            while (s < 16 && k[s] != kEmpty) ++s;
            if (s < 16) { k[s] = w.buildKeys[r]; rows[bkt * 16 + s] = uint32_t(r); break; }
         }
      }
   }
   ~BucketTable() { std::free(keys); std::free(rows); }
   BucketTable(const BucketTable&) = delete;
};

/// match / empty bitmasks of one bucket against key k
struct BucketMatch { uint32_t match, empty; };
inline BucketMatch compareBucket(const int32_t* bucket, int32_t k) {
#if defined(__x86_64__) && defined(__AVX512F__)
   const __m512i v = _mm512_load_si512(bucket);
   return {(uint32_t)_mm512_cmpeq_epi32_mask(v, _mm512_set1_epi32(k)),
           (uint32_t)_mm512_cmpeq_epi32_mask(v, _mm512_set1_epi32(kEmpty))};
#elif defined(__x86_64__) && defined(__AVX2__)
   const __m256i lo = _mm256_load_si256((const __m256i*)bucket), hi = _mm256_load_si256((const __m256i*)(bucket + 8));
   const __m256i vk = _mm256_set1_epi32(k), ve = _mm256_set1_epi32(kEmpty);
   auto m = [](__m256i x) { return (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(x)); };
   return {m(_mm256_cmpeq_epi32(lo, vk)) | m(_mm256_cmpeq_epi32(hi, vk)) << 8,
           m(_mm256_cmpeq_epi32(lo, ve)) | m(_mm256_cmpeq_epi32(hi, ve)) << 8};
#elif defined(__aarch64__)
   // NEON has no movemask: compare 4 x 4 lanes, pack to 16 bits via shifts
   const int32x4_t vk = vdupq_n_s32(k), ve = vdupq_n_s32(kEmpty);
   const uint32x4_t bits = {1, 2, 4, 8};
   uint32_t match = 0, empty = 0;
   for (int q = 0; q < 4; ++q) {
      const int32x4_t v = vld1q_s32(bucket + 4 * q);
      match |= vaddvq_u32(vandq_u32(vceqq_s32(v, vk), bits)) << (4 * q);
      empty |= vaddvq_u32(vandq_u32(vceqq_s32(v, ve), bits)) << (4 * q);
   }
   return {match, empty};
#else
   uint32_t match = 0, empty = 0;
   for (int s = 0; s < 16; ++s) match |= uint32_t(bucket[s] == k) << s, empty |= uint32_t(bucket[s] == kEmpty) << s;
   return {match, empty};
#endif
}

/// all build rows with key k: walk buckets from the hash's bucket until one has an empty slot
template <typename Emit>
inline void bucketProbe(const BucketTable& t, int32_t k, hash_t h, Emit&& emit) {
   for (uint64_t bkt = h & t.mask;; bkt = (bkt + 1) & t.mask) {
      const BucketMatch m = compareBucket(t.keys + bkt * 16, k);
      for (uint32_t mm = m.match; mm; mm &= mm - 1) emit(t.rows[bkt * 16 + __builtin_ctz(mm)]);
      if (m.empty) return;
   }
}

template <bool Prefetch>
Result probeJ3(BucketTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      for (size_t i = 0; i < n; ++i) {
         b.hashes[i] = hashKey(keys[base + i]);
         if (Prefetch) __builtin_prefetch(t.keys + (b.hashes[i] & t.mask) * 16);
      }
      for (size_t i = 0; i < n; ++i)
         bucketProbe(t, keys[base + i], b.hashes[i], [&](uint32_t row) {
            res.count += 1;
            res.checksum += mix(base + i, t.payload[row]);
         });
   }
   return res;
}

Result probeJ1J3(BucketTable& t, const Workload& w, Bufs& b) {
   Result res;
   const int32_t* keys = w.probeKeys.data();
   for (size_t base = 0; base < w.probeKeys.size(); base += kVec) {
      const size_t n = std::min(kVec, w.probeKeys.size() - base);
      size_t runs = 0;
      for (size_t i = 0; i < n; ++i)
         if (i == 0 || keys[base + i] != keys[base + i - 1]) b.runKeys[runs] = keys[base + i], b.runStart[runs++] = uint32_t(i);
      b.runStart[runs] = uint32_t(n);
      for (size_t r = 0; r < runs; ++r) b.hashes[r] = hashKey(b.runKeys[r]);
      for (size_t r = 0; r < runs; ++r)
         bucketProbe(t, b.runKeys[r], b.hashes[r], [&](uint32_t row) {
            for (uint32_t i = b.runStart[r]; i < b.runStart[r + 1]; ++i) {
               res.count += 1;
               res.checksum += mix(base + i, t.payload[row]);
            }
         });
   }
   return res;
}

template <typename F> double bestNs(F&& f, int reps, size_t rows) {
   double best = 1e300;
   for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      f();
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best / double(rows);
}

} // namespace

int main(int argc, char** argv) {
   double scale = 1.0;
   int reps = 5;
   std::string shapes = "q3,q5,q18,q9,q9j5,fk_all,dim";
   int opt;
   while ((opt = getopt(argc, argv, "s:r:q:")) != -1) {
      switch (opt) {
      case 's': scale = std::atof(optarg); break;
      case 'r': reps = std::atoi(optarg); break;
      case 'q': shapes = optarg; break;
      default: std::fprintf(stderr, "usage: %s [-s scale] [-r reps] [-q q3,q5,q18,q9,q9j5,fk_all,dim]\n", argv[0]); return 1;
      }
   }
   std::printf("shape,build_rows,probe_rows,matches,variant,ns_per_probe_row,speedup_vs_T0,check\n");
   int fails = 0;
   std::stringstream ss(shapes);
   std::string shape;
   auto bufs = std::make_unique<Bufs>();
   while (std::getline(ss, shape, ',')) {
      std::mt19937_64 rng(std::hash<std::string>{}(shape));
      const Workload w = makeWorkload(shape, scale, rng);
      ChainTable chain(w);
      BucketTable bucket(w);
      const Result ref = probeT0(chain, w, *bufs);
      struct V { const char* name; std::function<Result()> fn; };
      std::vector<V> vs = {
          {"T0 today", [&] { return probeT0(chain, w, *bufs); }},
          {"J1 run dedup", [&] { return probeJ1(chain, w, *bufs); }},
          {"J2 two-pass prefetch", [&] { return probeJ2(chain, w, *bufs); }},
          {"J2+J1a two-pass + gated dedup", [&] { return probeJ2J1a(chain, w, *bufs); }},
          {"J3 bucketized", [&] { return probeJ3<false>(bucket, w, *bufs); }},
          {"J3P bucketized+prefetch", [&] { return probeJ3<true>(bucket, w, *bufs); }},
          {"J1+J3", [&] { return probeJ1J3(bucket, w, *bufs); }},
#ifdef HAVE_J4
          {"J4 existing SIMD join", [&] { return probeJ4(chain, w, *bufs); }},
#endif
      };
      double t0ns = 0;
      for (auto& v : vs) {
         Result r;
         const double ns = bestNs([&] { r = v.fn(); }, reps, w.probeKeys.size());
         const bool ok = r == ref;
         fails += !ok;
         if (t0ns == 0) t0ns = ns;
         std::printf("%s,%zu,%zu,%llu,%s,%.3f,%.2f,%s\n", shape.c_str(), w.buildKeys.size(), w.probeKeys.size(),
                     (unsigned long long)ref.count, v.name, ns, t0ns / ns, ok ? "ok" : "MISMATCH");
         std::fflush(stdout);
      }
   }
#ifndef HAVE_J4
   std::fprintf(stderr, "run_joinbench: no AVX-512F, J4 (existing SIMD join) skipped\n");
#endif
   return fails != 0;
}
