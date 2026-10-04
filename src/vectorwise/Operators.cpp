#include "vectorwise/Operators.hpp"
#include "vectorwise/Primitives.hpp"
#include "common/Compat.hpp"
#include "common/runtime/CacheInfo.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Hashmap.hpp"
#include "common/runtime/SIMD.hpp"
#include "vectorwise/SimdCrc.hpp"
#include "vectorwise/SimdHash.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/avx512.h>

namespace vectorwise {

using runtime::barrier;

size_t Select::next() {
   while (true) {
      auto n = child->next();
      if (n == EndOfStream) return EndOfStream;
      n = condition->evaluate(n);
      if (n > 0) return n;
   }
}

size_t Project::next() {
   auto n = child->next();
   if (n == EndOfStream) return EndOfStream;
   for (auto& expression : expressions) expression->evaluate(n);
   return n;
}

size_t FixedAggr::next() {
   if (!consumed) {
      size_t found = 0;
      for (auto n = child->next(); n != EndOfStream; n = child->next()) {
         found = aggregates.evaluate(n);
      }
      consumed = true;
      return found;
   } else {
      return EndOfStream;
   }
}

Scan::Scan(Shared& s, size_t n, size_t v)
    : shared(s), needsInit(true), currentChunk(0), lastOffset(0), nrTuples(n),
      vecSize(v) {
   scanChunkSize = 1;
#ifndef CFG_MORSEL_SIZE
#define CFG_MORSEL_SIZE 10000
#endif
   size_t scanMorselSize = CFG_MORSEL_SIZE;
   if (vecSize < scanMorselSize) scanChunkSize = scanMorselSize / vecSize + 1;
   vecInChunk = scanChunkSize;
}

void Scan::addConsumer(void** colPtr, size_t typeSize) {
   consumers.emplace_back(colPtr, vecSize * typeSize);
}

size_t Scan::next() {
   auto step = 1;

   if (vecInChunk == scanChunkSize) {
      auto prevChunk = currentChunk;
      currentChunk = shared.pos.fetch_add(1);
      auto chunkSkip = currentChunk - prevChunk;
      if (needsInit) {
         step = chunkSkip * scanChunkSize;
         needsInit = false;
      } else {
         chunkSkip -= 1;
         step = chunkSkip * scanChunkSize + 1;
      }
      vecInChunk = 0;
   }

   auto nextBegin = lastOffset + step * vecSize;
   if (nextBegin >= nrTuples) return EndOfStream;
   auto nextBatchSize = std::min(nrTuples - nextBegin, vecSize);
   for (auto& cons : consumers)
      *cons.first = (void*)(*(uint8_t**)cons.first + step * cons.second);
   lastOffset = nextBegin;
   vecInChunk++;
   return nextBatchSize;
}

ResultWriter::Input::Input(void* d, size_t size,
                           runtime::BlockRelation::Attribute attr)
    : data(d), elementSize(size), attribute(attr) {}

ResultWriter::ResultWriter(Shared& s)
    : shared(s), currentBlock(nullptr, nullptr) {}

size_t ResultWriter::next() {
   size_t found = 0;
   for (pos_t n = child->next(); n != EndOfStream; n = child->next()) {
      found += n;
      // assure that enough space is available in current block to fit result of
      // all buffers
      if (currentBlock.spaceRemaining() < n)
         currentBlock = shared.result->result->createBlock(n);
      auto blockSize = currentBlock.size();
      for (const auto& input : inputs)
         // copy data from intermediate buffers into result relation
         std::memcpy(addBytes(currentBlock.data(input.attribute),
                              input.elementSize * blockSize),
                     input.data, n * input.elementSize);
      // update result relation size
      currentBlock.addedElements(n);
   }
   return found;
}

pos_t Hashjoin::joinAll() {
   size_t found = 0;
   // perform continuation
   for (auto entry = cont.buildMatch; entry != shared.ht.end();
        entry = entry->next) {
      if (entry->hash == cont.probeHash) {
         buildMatches[found] = entry;
         probeMatches[found++] = cont.nextProbe;
         if (found == batchSize) {
            // output buffers are full, save state for continuation
            cont.buildMatch = entry->next;
            return batchSize;
         }
      }
   }
   if (cont.buildMatch != shared.ht.end()) cont.nextProbe++;
   for (size_t i = cont.nextProbe, end = cont.numProbes; i < end; ++i) {
      auto hash = probeHashes[i];
      for (auto entry = shared.ht.find_chain_tagged(hash);
           entry != shared.ht.end(); entry = entry->next) {
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = i;
            if (found == batchSize && (entry->next || i + 1 < end)) {
               // output buffers are full, save state for continuation
               cont.buildMatch = entry->next;
               cont.probeHash = hash;
               cont.nextProbe = i;
               return batchSize;
            }
         }
      }
   }
   cont.buildMatch = shared.ht.end();
   cont.nextProbe = cont.numProbes;
   return found;
}

// VW_JOIN_PREFETCH: two-stage software prefetch over a probe vector whose
// hashes are all known up front. Stage 1 prefetches the directory slot 2*D
// probes ahead; stage 2 (D ahead) reads that slot, now cached, and prefetches
// the first chain entry when the tag filter matches.
#ifdef VW_JOIN_PREFETCH
#ifndef VW_JOIN_PREFETCH_DIST
#define VW_JOIN_PREFETCH_DIST 16
#endif
namespace {
constexpr size_t kJoinPfDist = VW_JOIN_PREFETCH_DIST;
static_assert(kJoinPfDist > 0, "VW_JOIN_PREFETCH_DIST must be positive");

inline void joinPrefetchWarmup(runtime::Hashmap& ht,
                               const runtime::Hashmap::hash_t* hashes,
                               size_t n) {
   for (size_t i = 0, e = std::min(n, 2 * kJoinPfDist); i < e; ++i)
      ht.prefetch_slot(hashes[i]);
   for (size_t i = 0, e = std::min(n, kJoinPfDist); i < e; ++i)
      ht.prefetch_chain_tagged(hashes[i]);
}

inline void joinPrefetchStep(runtime::Hashmap& ht,
                             const runtime::Hashmap::hash_t* hashes, size_t i,
                             size_t n) {
   if (i + 2 * kJoinPfDist < n) ht.prefetch_slot(hashes[i + 2 * kJoinPfDist]);
   if (i + kJoinPfDist < n) ht.prefetch_chain_tagged(hashes[i + kJoinPfDist]);
}
} // namespace
#define VW_JOIN_PF_WARMUP(ht, hashes, n) joinPrefetchWarmup(ht, hashes, n)
#define VW_JOIN_PF_STEP(ht, hashes, i, n) joinPrefetchStep(ht, hashes, i, n)
#else
#define VW_JOIN_PF_WARMUP(ht, hashes, n) ((void)0)
#define VW_JOIN_PF_STEP(ht, hashes, i, n) ((void)0)
#endif

// VW_JOIN_TWOPHASE: the first pass of joinAllParallel / joinSelParallel in
// three loops over the probe vector (group prefetching), so the directory
// and entry misses of the whole vector overlap instead of one per probe:
//   A  prefetch every probe's directory slot
//   B  read the slots (now cached), tag-filter, prefetch the first entries;
//      candidates go to the followup buffers
//   C  compare hashes, emit matches, queue chain successors; compacts the
//      followup buffers in place (write index <= read index)
// Directories below VW_JOIN_TWOPHASE_MIN_SLOTS slots stay cache-resident, and
// there the extra loops only cost (TPC-H Q18's 57-order build: 0.9x in
// run_joinbench on every machine), so those joins keep the one-loop path.
#ifdef VW_JOIN_TWOPHASE
#ifndef VW_JOIN_TWOPHASE_MIN_SLOTS
#define VW_JOIN_TWOPHASE_MIN_SLOTS 4096
#endif
namespace {
constexpr size_t kTwoPhaseMinSlots = VW_JOIN_TWOPHASE_MIN_SLOTS;

inline bool useTwoPhase(const runtime::Hashmap& ht) {
   return ht.mask + 1 >= kTwoPhaseMinSlots;
}

/// Matches of probes 0..n-1 into buildMatches / probeMatches (probe id(i)),
/// chain successors into followupIds / followupEntries from index 0; returns
/// the match count, sets followupWrite to the successor count
template <typename Id>
inline size_t joinTwoPhase(runtime::Hashmap& ht,
                           const runtime::Hashmap::hash_t* hashes, size_t n,
                           runtime::Hashmap::EntryHeader** buildMatches,
                           pos_t* probeMatches, pos_t* followupIds,
                           runtime::Hashmap::EntryHeader** followupEntries,
                           pos_t& followupWrite, Id id) {
   for (size_t i = 0; i < n; ++i) ht.prefetch_slot(hashes[i]);
   size_t candidates = 0;
   for (size_t i = 0; i < n; ++i) {
      auto entry = ht.find_chain_tagged(hashes[i]);
      if (entry != ht.end()) {
         __builtin_prefetch(entry, 0, 3);
         followupEntries[candidates] = entry;
         followupIds[candidates++] = i;
      }
   }
   size_t found = 0;
   pos_t write = 0;
   for (size_t j = 0; j < candidates; ++j) {
      auto entry = followupEntries[j];
      const pos_t i = followupIds[j];
      if (entry->hash == hashes[i]) {
         buildMatches[found] = entry;
         probeMatches[found++] = id(i);
      }
      if (entry->next != ht.end()) {
         followupIds[write] = i;
         followupEntries[write++] = entry->next;
      }
   }
   followupWrite = write;
   return found;
}
} // namespace
#endif

pos_t Hashjoin::joinAllParallel() {
   size_t found = 0;
   auto followup = contCon.followup;
   auto followupWrite = contCon.followupWrite;

   if (followup == followupWrite) {
      VW_JOIN_PF_WARMUP(shared.ht, probeHashes, cont.numProbes);
#ifdef VW_JOIN_TWOPHASE
      if (useTwoPhase(shared.ht))
         found = joinTwoPhase(shared.ht, probeHashes, cont.numProbes,
                              buildMatches, probeMatches, followupIds,
                              followupEntries, followupWrite,
                              [](pos_t i) { return i; });
      else
#endif
#ifdef VW_JOIN_BLOOM
      // only the probes that passed the Bloom filter (all, when it is off)
      const pos_t* cand = bloomSel.data();
      const bool bf = bloomOn;
      for (size_t k = 0, end = bf ? bloomCount : cont.numProbes; k < end; ++k) {
         const size_t i = bf ? cand[k] : k;
#else
      for (size_t i = 0, end = cont.numProbes; i < end; ++i) {
#endif
         VW_JOIN_PF_STEP(shared.ht, probeHashes, i, end);
         auto hash = probeHashes[i];
         auto entry = shared.ht.find_chain_tagged(hash);
         if (entry != shared.ht.end()) {
            if (entry->hash == hash) {
               buildMatches[found] = entry;
               probeMatches[found] = i;
               found += 1;
            }
            if (entry->next != shared.ht.end()) {
               followupIds[followupWrite] = i;
               followupEntries[followupWrite] = entry->next;
               followupWrite += 1;
            }
         }
      }
   }

   followupWrite %= followupBufferSize;

   while (followup != followupWrite) {
      auto remainingSpace = batchSize - found;
      auto nrFollowups = followup <= followupWrite
                             ? followupWrite - followup
                             : followupBufferSize - (followup - followupWrite);
      // std::cout << "nrFollowups: " << nrFollowups << "\n";
      auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
      for (size_t j = 0; j < fittingElements; ++j) {
         size_t i = followupIds[followup];
         auto entry = followupEntries[followup];
         // followup = (followup + 1) % followupBufferSize;
         followup = (followup + 1);
         if (followup == followupBufferSize) followup = 0;
         auto hash = probeHashes[i];
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = i;
         }
         if (entry->next != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry->next;
            followupWrite = (followupWrite + 1) % followupBufferSize;
         }
      }
      if (fittingElements < nrFollowups) {
         // continuation
         contCon.followupWrite = followupWrite;
         contCon.followup = followup;
         return found;
      }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followup = 0;
   contCon.followupWrite = 0;
   return found;
}

#ifndef VW_POS_16
pos_t Hashjoin::joinAllSIMD() {
   size_t found = 0;
   auto followup = contCon.followup;
   auto followupWrite = contCon.followupWrite;

   if (followup == followupWrite) {

#ifdef __AVX512F__ // if AVX 512 available, use it!
#if HASH_SIZE == 32
      size_t rest = cont.numProbes % 8;
      auto ids =
          _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
      for (size_t i = 0, end = cont.numProbes - rest; i < end; i += 8) {

         // load hashes
         // auto hash = probeHashes[i];
         // Vec8u hashes(probeHashes + i);
         auto hashDense = _mm256_loadu_si256((const __m256i*)(probeHashes + i));
         Vec8u hashes = _mm512_cvtepu32_epi64(hashDense);
         // find entry pointers in ht
         Vec8uM entries = shared.ht.find_chain_tagged(hashes);
         // load entry hashes
         auto entryHashes = _mm512_mask_i64gather_epi32(
             hashDense, entries.mask,
             entries.vec +
                 Vec8u(offsetof(decltype(shared.ht)::EntryHeader, hash)),
             nullptr, 1);
         {
            // Check if hashes match
            __mmask8 hashesEq = _mm512_mask_cmpeq_epi32_mask(
                entries.mask, _mm512_castsi256_si512(entryHashes),
                _mm512_castsi256_si512(hashDense));
            // write pointers
            _mm512_mask_compressstoreu_epi64(buildMatches + found, hashesEq,
                                             entries.vec);
            static_assert(sizeof(pos_t) == 4,
                          "SIMD join assumes sizeof(pos_t) is 4"); // change the
                                                                   // types for
                                                                   // probeSels
                                                                   // if this
                                                                   // fails
            // write selection
            _mm512_mask_compressstoreu_epi32(probeMatches + found, hashesEq,
                                             ids);
            found += __builtin_popcount(hashesEq);
         }

         {
            // write continuations
            static_assert(offsetof(decltype(shared.ht)::EntryHeader, next) == 0,
                          "Hash is expected to be in first position");
            Vec8u nextPtrs = _mm512_mask_i64gather_epi64(
                entries.vec, entries.mask, entries.vec, nullptr, 1);
            __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(
                entries.mask, nextPtrs, Vec8u(uint64_t(shared.ht.end())));
            if (hasNext) {
               // write pointers
               _mm512_mask_compressstoreu_epi64(followupEntries + followupWrite,
                                                hasNext, nextPtrs);
               static_assert(
                   sizeof(pos_t) == 4,
                   "SIMD join assumes sizeof(pos_t) is 4"); // change the types
                                                            // for probeSels if
                                                            // this fails
               // write selection
               _mm512_mask_compressstoreu_epi32(followupIds + followupWrite,
                                                hasNext, ids);
               followupWrite += __builtin_popcount(hasNext);
            }
            ids = _mm512_add_epi32(ids, _mm512_set1_epi32(8));
         }
      }
#else
      size_t rest = cont.numProbes % 8;
      // auto ids = _mm256_set_epi32(7,6,5,4,3,2,1,0);
      auto ids =
          _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
      for (size_t i = 0, end = cont.numProbes - rest; i < end; i += 8) {

         // load hashes
         // auto hash = probeHashes[i];
         Vec8u hashes(probeHashes + i);
         // find entry pointers in ht
         Vec8uM entries = shared.ht.find_chain_tagged(hashes);
         // load entry hashes
         Vec8u entryHashes = _mm512_mask_i64gather_epi64(
             entries.vec, entries.mask,
             entries.vec +
                 Vec8u(offsetof(decltype(shared.ht)::EntryHeader, hash)),
             nullptr, 1);
         {
            // Check if hashes match
            __mmask8 hashesEq =
                _mm512_mask_cmpeq_epi64_mask(entries.mask, entryHashes, hashes);
            // write pointers
            _mm512_mask_compressstoreu_epi64(buildMatches + found, hashesEq,
                                             entries.vec);
            static_assert(sizeof(pos_t) == 4,
                          "SIMD join assumes sizeof(pos_t) is 4"); // change the
                                                                   // types for
                                                                   // probeSels
                                                                   // if this
                                                                   // fails
            // write selection
            _mm512_mask_compressstoreu_epi32(probeMatches + found, hashesEq,
                                             ids);
            found += __builtin_popcount(hashesEq);
         }

         {
            // write continuations
            static_assert(offsetof(decltype(shared.ht)::EntryHeader, next) == 0,
                          "Hash is expected to be in first position");
            Vec8u nextPtrs = _mm512_mask_i64gather_epi64(
                entries.vec, entries.mask, entries.vec, nullptr, 1);
            __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(
                entries.mask, nextPtrs, Vec8u(uint64_t(shared.ht.end())));
            if (hasNext) {
               // write pointers
               _mm512_mask_compressstoreu_epi64(followupEntries + followupWrite,
                                                hasNext, nextPtrs);
               static_assert(
                   sizeof(pos_t) == 4,
                   "SIMD join assumes sizeof(pos_t) is 4"); // change the types
                                                            // for probeSels if
                                                            // this fails
               // write selection
               _mm512_mask_compressstoreu_epi32(followupIds + followupWrite,
                                                hasNext, ids);
               followupWrite += __builtin_popcount(hasNext);
            }
            ids = _mm512_add_epi32(ids, _mm512_set1_epi32(8));
         }
      }
#endif // hash size
#else
      const size_t rest = cont.numProbes;
#endif
      for (size_t i = cont.numProbes - rest, end = cont.numProbes; i < end;
           ++i) {
         auto hash = probeHashes[i];
         auto entry = shared.ht.find_chain_tagged(hash);
         if (entry != shared.ht.end()) {
            if (entry->hash == hash) {
               buildMatches[found] = entry;
               probeMatches[found] = i;
               found += 1;
            }
            if (entry->next != shared.ht.end()) {
               followupIds[followupWrite] = i;
               followupEntries[followupWrite] = entry->next;
               followupWrite += 1;
            }
         }
      }
   }

   followupWrite %= followupBufferSize;

   while (followup != followupWrite) {
      auto remainingSpace = batchSize - found;
      auto nrFollowups = followup <= followupWrite
                             ? followupWrite - followup
                             : followupBufferSize - (followup - followupWrite);
      auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
      for (size_t j = 0; j < fittingElements; ++j) {
         size_t i = followupIds[followup];
         auto entry = followupEntries[followup];
         followup = (followup + 1);
         if (followup == followupBufferSize) followup = 0;
         auto hash = probeHashes[i];
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = i;
         }
         if (entry->next != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry->next;
            followupWrite = (followupWrite + 1) % followupBufferSize;
         }
      }
      if (fittingElements < nrFollowups) {
         // continuation
         contCon.followupWrite = followupWrite;
         contCon.followup = followup;
         return found;
      }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followup = 0;
   contCon.followupWrite = 0;
   return found;
}
#endif // !VW_POS_16

pos_t Hashjoin::joinSel() {
   size_t found = 0;
   // perform continuation
   for (auto entry = cont.buildMatch; entry != shared.ht.end();
        entry = entry->next) {
      if (entry->hash == cont.probeHash) {
         buildMatches[found] = entry;
         probeMatches[found++] = probeSel[cont.nextProbe];
         if (found == batchSize) {
            // output buffers are full, save state for continuation
            cont.buildMatch = entry->next;
            return batchSize;
         }
      }
   }
   if (cont.buildMatch != shared.ht.end()) cont.nextProbe++;
   for (size_t i = cont.nextProbe, end = cont.numProbes; i < end; ++i) {
      auto hash = probeHashes[i];
      for (auto entry = shared.ht.find_chain_tagged(hash);
           entry != shared.ht.end(); entry = entry->next) {
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = probeSel[i];
            if (found == batchSize && (entry->next || i + 1 < end)) {
               // output buffers are full, save state for continuation
               cont.buildMatch = entry->next;
               cont.probeHash = hash;
               cont.nextProbe = i;
               return batchSize;
            }
         }
      }
   }
   cont.buildMatch = shared.ht.end();
   cont.nextProbe = cont.numProbes;
   return found;
}

pos_t Hashjoin::joinSelParallel() {
   size_t found = 0;
   auto followup = contCon.followup;
   auto followupWrite = contCon.followupWrite;

   if (followup == followupWrite) {
      VW_JOIN_PF_WARMUP(shared.ht, probeHashes, cont.numProbes);
#ifdef VW_JOIN_TWOPHASE
      if (useTwoPhase(shared.ht))
         found = joinTwoPhase(shared.ht, probeHashes, cont.numProbes,
                              buildMatches, probeMatches, followupIds,
                              followupEntries, followupWrite,
                              [sel = probeSel](pos_t i) { return sel[i]; });
      else
#endif
#ifdef VW_JOIN_BLOOM
      // only the probes that passed the Bloom filter (all, when it is off)
      const pos_t* cand = bloomSel.data();
      const bool bf = bloomOn;
      for (size_t k = 0, end = bf ? bloomCount : cont.numProbes; k < end; ++k) {
         const size_t i = bf ? cand[k] : k;
#else
      for (size_t i = 0, end = cont.numProbes; i < end; ++i) {
#endif
         VW_JOIN_PF_STEP(shared.ht, probeHashes, i, end);
         auto hash = probeHashes[i];
         auto entry = shared.ht.find_chain_tagged(hash);
         if (entry != shared.ht.end()) {
            if (entry->hash == hash) {
               buildMatches[found] = entry;
               probeMatches[found] = probeSel[i];
               found += 1;
            }
            if (entry->next != shared.ht.end()) {
               followupIds[followupWrite] = i;
               followupEntries[followupWrite] = entry->next;
               followupWrite += 1;
            }
         }
      }
   }

   followupWrite %= followupBufferSize;

   while (followup != followupWrite) {
      auto remainingSpace = batchSize - found;
      auto nrFollowups = followup <= followupWrite
                             ? followupWrite - followup
                             : followupBufferSize - (followup - followupWrite);
      auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
      for (size_t j = 0; j < fittingElements; ++j) {
         size_t i = followupIds[followup];
         auto entry = followupEntries[followup];
         followup = (followup + 1) % followupBufferSize;
         auto hash = probeHashes[i];
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = probeSel[i];
         }
         if (entry->next != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry->next;
            followupWrite = (followupWrite + 1) % followupBufferSize;
         }
      }
      if (fittingElements < nrFollowups) {
         // continuation
         contCon.followupWrite = followupWrite;
         contCon.followup = followup;
         return found;
      }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followup = 0;
   contCon.followupWrite = 0;
   return found;
}

#ifndef VW_POS_16
pos_t Hashjoin::joinSelSIMD() {
   size_t found = 0;
   auto followup = contCon.followup;
   auto followupWrite = contCon.followupWrite;

   if (followup == followupWrite) {

#ifdef __AVX512F__ // if AVX 512 available, use it!
#if HASH_SIZE == 32
      size_t rest = cont.numProbes % 8;
      auto ids =
          _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
      for (size_t i = 0, end = cont.numProbes - rest; i < end; i += 8) {

         // load hashes
         // auto hash = probeHashes[i];
         auto hashDense = _mm256_loadu_si256((const __m256i*)(probeHashes + i));
         Vec8u hashes = _mm512_cvtepu32_epi64(hashDense);
         // Vec8u hashes(probeHashes + i);
         // find entry pointers in ht
         Vec8uM entries = shared.ht.find_chain_tagged(hashes);
         // load entry hashes
         Vec8u hashPtrs =
             entries.vec +
             Vec8u(offsetof(decltype(shared.ht)::EntryHeader, hash));
         auto entryHashes = _mm512_mask_i64gather_epi32(hashDense, entries.mask,
                                                        hashPtrs, nullptr, 1);
         {
            // Check if hashes match
            __mmask8 hashesEq = _mm512_mask_cmpeq_epi32_mask(
                entries.mask, _mm512_castsi256_si512(entryHashes),
                _mm512_castsi256_si512(hashDense));
            // write pointers
            _mm512_mask_compressstoreu_epi64(buildMatches + found, hashesEq,
                                             entries.vec);
            static_assert(sizeof(pos_t) == 4,
                          "SIMD join assumes sizeof(pos_t) is 4"); // change the
                                                                   // types for
                                                                   // probeSels
                                                                   // if this
                                                                   // fails
            // write selection
            __m512i probeSels = _mm512_loadu_si512(probeSel + i);
            _mm512_mask_compressstoreu_epi32(probeMatches + found, hashesEq,
                                             probeSels);
            found += __builtin_popcount(hashesEq);
         }

         {
            // write continuations
            static_assert(offsetof(decltype(shared.ht)::EntryHeader, next) == 0,
                          "Hash is expected to be in first position");
            Vec8u nextPtrs = _mm512_mask_i64gather_epi64(
                entries.vec, entries.mask, entries.vec, nullptr, 1);
            __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(
                entries.mask, nextPtrs, Vec8u(uint64_t(shared.ht.end())));
            if (hasNext) {
               // write pointers
               _mm512_mask_compressstoreu_epi64(followupEntries + followupWrite,
                                                hasNext, nextPtrs);
               static_assert(
                   sizeof(pos_t) == 4,
                   "SIMD join assumes sizeof(pos_t) is 4"); // change the types
                                                            // for probeSels if
                                                            // this fails
               // write selection
               _mm512_mask_compressstoreu_epi32(followupIds + followupWrite,
                                                hasNext, ids);
               followupWrite += __builtin_popcount(hasNext);
            }
            ids = _mm512_add_epi32(ids, _mm512_set1_epi32(8));
         }
      }
#else
      size_t rest = cont.numProbes % 8;
      auto ids =
          _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
      for (size_t i = 0, end = cont.numProbes - rest; i < end; i += 8) {

         // load hashes
         // auto hash = probeHashes[i];
         Vec8u hashes(probeHashes + i);
         // find entry pointers in ht
         Vec8uM entries = shared.ht.find_chain_tagged(hashes);
         // load entry hashes
         Vec8u hashPtrs =
             entries.vec +
             Vec8u(offsetof(decltype(shared.ht)::EntryHeader, hash));
         Vec8u entryHashes = _mm512_mask_i64gather_epi64(hashPtrs, entries.mask,
                                                         hashPtrs, nullptr, 1);
         {
            // Check if hashes match
            __mmask8 hashesEq =
                _mm512_mask_cmpeq_epi64_mask(entries.mask, entryHashes, hashes);
            // write pointers
            _mm512_mask_compressstoreu_epi64(buildMatches + found, hashesEq,
                                             entries.vec);
            static_assert(sizeof(pos_t) == 4,
                          "SIMD join assumes sizeof(pos_t) is 4"); // change the
                                                                   // types for
                                                                   // probeSels
                                                                   // if this
                                                                   // fails
            // write selection
            __m512i probeSels = _mm512_loadu_si512(probeSel + i);
            _mm512_mask_compressstoreu_epi32(probeMatches + found, hashesEq,
                                             probeSels);
            found += __builtin_popcount(hashesEq);
         }

         {
            // write continuations
            static_assert(offsetof(decltype(shared.ht)::EntryHeader, next) == 0,
                          "Hash is expected to be in first position");
            Vec8u nextPtrs = _mm512_mask_i64gather_epi64(
                entries.vec, entries.mask, entries.vec, nullptr, 1);
            __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(
                entries.mask, nextPtrs, Vec8u(uint64_t(shared.ht.end())));
            if (hasNext) {
               // write pointers
               _mm512_mask_compressstoreu_epi64(followupEntries + followupWrite,
                                                hasNext, nextPtrs);
               static_assert(
                   sizeof(pos_t) == 4,
                   "SIMD join assumes sizeof(pos_t) is 4"); // change the types
                                                            // for probeSels if
                                                            // this fails
               // write selection
               _mm512_mask_compressstoreu_epi32(followupIds + followupWrite,
                                                hasNext, ids);
               followupWrite += __builtin_popcount(hasNext);
            }
            ids = _mm512_add_epi32(ids, _mm512_set1_epi32(8));
         }
      }
#endif // hash size
#else
      const size_t rest = cont.numProbes;
#endif
      for (size_t i = cont.numProbes - rest, end = cont.numProbes; i < end;
           ++i) {
         auto hash = probeHashes[i];
         auto entry = shared.ht.find_chain_tagged(hash);
         if (entry != shared.ht.end()) {
            if (entry->hash == hash) {
               buildMatches[found] = entry;
               probeMatches[found] = probeSel[i];
               found += 1;
            }
            if (entry->next != shared.ht.end()) {
               followupIds[followupWrite] = i;
               followupEntries[followupWrite] = entry->next;
               followupWrite += 1;
            }
         }
      }
   }

   followupWrite %= followupBufferSize;

   while (followup != followupWrite) {
      auto remainingSpace = batchSize - found;
      auto nrFollowups = followup <= followupWrite
                             ? followupWrite - followup
                             : followupBufferSize - (followup - followupWrite);
      auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
      for (size_t j = 0; j < fittingElements; ++j) {
         size_t i = followupIds[followup];
         auto entry = followupEntries[followup];
         followup = (followup + 1) % followupBufferSize;
         auto hash = probeHashes[i];
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = probeSel[i];
         }
         if (entry->next != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry->next;
            followupWrite = (followupWrite + 1) % followupBufferSize;
         }
      }
      if (fittingElements < nrFollowups) {
         // continuation
         contCon.followupWrite = followupWrite;
         contCon.followup = followup;
         return found;
      }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followup = 0;
   contCon.followupWrite = 0;
   return found;
}
#endif // !VW_POS_16

#if (defined(VW_JOIN_BLOOM) || defined(VW_NEW_JOIN)) && defined(__AVX512F__)
// Compress-store via a register compress plus a masked store: the memory form
// (vpcompressd/q to memory) is microcoded and very slow on Zen 4 (the reason
// VW_SIMD_SEL_COMPRESS=reg exists). Writes exactly popcount(m) elements.
namespace {
inline __attribute__((always_inline)) void compressStore32(void* dst, __mmask8 m,
                                                           __m512i v) {
   const __m512i c = _mm512_maskz_compress_epi32((__mmask16)m, v);
   _mm512_mask_storeu_epi32(dst, (__mmask16)((1u << __builtin_popcount(m)) - 1),
                            c);
}
inline __attribute__((always_inline)) void compressStore64(void* dst, __mmask8 m,
                                                           __m512i v) {
   const __m512i c = _mm512_maskz_compress_epi64(m, v);
   _mm512_mask_storeu_epi64(dst, (__mmask8)((1u << __builtin_popcount(m)) - 1),
                            c);
}
} // namespace
#endif

#ifdef VW_JOIN_SEMI
// VW_JOIN_SEMI: a Hashjoin the plan marks semi (HashJoinBuilder::semi())
// only asks whether a probe key exists in the build side. Q18's orders join
// is a semi join in the SQL (IN subquery); Q3 customer, Q5 region and Q9 part
// are inner joins on a primary key whose build side contributes no columns,
// rewritten to semi joins as Hyper's plans do (Hashset::contains). Unmarked
// joins always stay inner joins. Inner and semi join agree only because the
// marked joins' build keys are unique, the assumption semi() documents: this
// bitmap emits a probe row at most once, the hash path once per match.
// For one int32 key per side,
// the build keys' range [min, max] gets an exact bitmap (built after the
// hash table, at most 2^27 keys = 16 MB; Q3's customer keys ~19 KB, Q18's 57
// orders over ~6M order keys ~750 KB); a probe vector is then filtered with
// a range check and one bit test per key (AVX-512: 16 keys per step) into
// probeMatches. The probe hash pass, directory, chains, key equality and
// gather are all skipped. Other joins keep the hash path.
void Hashjoin::semiRange() {
   int64_t lo = INT64_MAX, hi = INT64_MIN;
   for (auto& block : allocations) {
      auto e = static_cast<char*>(block.first);
      for (size_t i = 0; i < block.second; ++i, e += ht_entry_size) {
         int32_t k;
         std::memcpy(&k, e + semiKeyOffset, sizeof(k));
         lo = std::min<int64_t>(lo, k);
         hi = std::max<int64_t>(hi, k);
      }
   }
   for (int64_t cur = shared.semiMin.load();
        lo < cur && !shared.semiMin.compare_exchange_weak(cur, lo);) {}
   for (int64_t cur = shared.semiMax.load();
        hi > cur && !shared.semiMax.compare_exchange_weak(cur, hi);) {}
}

void Hashjoin::semiSetBits() {
   uint32_t* bits = shared.semiBits.get();
   const int64_t lo = shared.semiMin.load();
   const bool concurrent = runtime::this_worker->group->size > 1;
   for (auto& block : allocations) {
      auto e = static_cast<char*>(block.first);
      for (size_t i = 0; i < block.second; ++i, e += ht_entry_size) {
         int32_t k;
         std::memcpy(&k, e + semiKeyOffset, sizeof(k));
         const uint64_t d = uint64_t(int64_t(k) - lo);
         const uint32_t bit = uint32_t(1) << (d & 31);
         if (concurrent)
            __atomic_fetch_or(&bits[d >> 5], bit, __ATOMIC_RELAXED);
         else
            bits[d >> 5] |= bit;
      }
   }
}

pos_t Hashjoin::semiProbe(size_t n) {
   const bool sel = semiProbeSel != nullptr;
   const int32_t* keys = static_cast<const int32_t*>(
       sel ? semiProbeSel->param2 : semiProbeDense->param1);
   const pos_t* ksel =
       sel ? static_cast<const pos_t*>(semiProbeSel->outputSelectionV) : nullptr;
   const uint32_t* bits = shared.semiBits.get();
   const int64_t lo = shared.semiMin.load();
   const uint64_t span = uint64_t(shared.semiMax.load() - lo); // last offset
   pos_t found = 0;
   size_t i = 0;
#if defined(__AVX512F__) && !defined(VW_POS_16)
   if (lo >= INT32_MIN && lo <= INT32_MAX) {
      const __m512i vlo = _mm512_set1_epi32(int32_t(lo));
      const __m512i vspan = _mm512_set1_epi32(int32_t(span));
      const __m512i b31 = _mm512_set1_epi32(31);
      const __m512i one = _mm512_set1_epi32(1);
      const __m512i lane =
          _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
      for (; i + 16 <= n; i += 16) {
         const __m512i pos =
             sel ? _mm512_loadu_si512(ksel + i)
                 : _mm512_add_epi32(_mm512_set1_epi32(int32_t(i)), lane);
         const __m512i k = sel ? _mm512_i32gather_epi32(pos, keys, 4)
                               : _mm512_loadu_si512(keys + i);
         // key - min, as unsigned: out-of-range keys wrap above span
         const __m512i d = _mm512_sub_epi32(k, vlo);
         const __mmask16 in = _mm512_cmple_epu32_mask(d, vspan);
         const __m512i w = _mm512_mask_i32gather_epi32(
             _mm512_setzero_si512(), in, _mm512_srli_epi32(d, 5), bits, 4);
         const __mmask16 hit = _mm512_mask_test_epi32_mask(
             in, w, _mm512_sllv_epi32(one, _mm512_and_si512(d, b31)));
         // output: probe position (join_sel: the selected position)
         const __m512i c = _mm512_maskz_compress_epi32(hit, pos);
         _mm512_mask_storeu_epi32(probeMatches + found,
                                  (__mmask16)((1u << __builtin_popcount(hit)) - 1),
                                  c);
         found += __builtin_popcount(hit);
      }
   }
#endif
   for (; i < n; ++i) {
      const pos_t p = sel ? ksel[i] : pos_t(i);
      const uint64_t d = uint64_t(int64_t(keys[p]) - lo);
      if (d <= span && (bits[d >> 5] >> (d & 31) & 1)) probeMatches[found++] = p;
   }
   return found;
}
#endif // VW_JOIN_SEMI

#ifdef VW_JOIN_BLOOM
// VW_JOIN_BLOOM: a Bloom filter owned by this VectorWise join, in front of
// the shared runtime::Hashmap (unchanged, so Hyper is unaffected). Most
// probes of TPC-H Q3/Q5/Q9-style joins miss (90-99%); the directory tag
// still costs a random directory load per probe, and a false tag pass an
// entry load. The filter is ~VW_JOIN_BLOOM_BITS bits per build key (16:
// 300 KB for Q3's 150k orders, L2-resident) and answers most misses from
// one word: register-blocked, a key's 4 bits all lie in one 64-bit word.
//   word = hash bits 40.. & mask        bits = hash bits 0..23, 6 per bit
// (independent bit groups). Once per probe vector the probe hashes are
// tested 8 at a time (AVX-512: gather 8 words, 4 variable shifts, compare)
// and the passing probes are compressed into bloomSel / bloomHashes; the
// join's first pass then visits only those. A vector where more than half
// the probes pass (mostly-hitting joins, e.g. Q9's) turns the filter off for
// the next 31 vectors.
namespace {
inline uint64_t bloomBits(runtime::Hashmap::hash_t h) {
   return (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
          (uint64_t(1) << ((h >> 12) & 63)) | (uint64_t(1) << ((h >> 18) & 63));
}
} // namespace

void Hashjoin::bloomInsert() {
   using EH = runtime::Hashmap::EntryHeader;
   uint64_t* bloom = shared.bloom.get();
   const uint64_t mask = shared.bloomMask;
   const bool concurrent = runtime::this_worker->group->size > 1;
   for (auto& block : allocations) {
      auto e = reinterpret_cast<EH*>(block.first);
      for (size_t i = 0; i < block.second; ++i, e = addBytes(e, ht_entry_size)) {
         const auto h = e->hash;
         uint64_t* w = &bloom[(h >> 40) & mask];
         if (concurrent)
            __atomic_fetch_or(w, bloomBits(h), __ATOMIC_RELAXED);
         else
            *w |= bloomBits(h);
      }
   }
}

void Hashjoin::bloomFilter(size_t n) {
   if (bloomSel.size() < n) {
      bloomSel.resize(n);
      bloomHashes.resize(n);
   }
   const uint64_t* bloom = shared.bloom.get();
   const uint64_t mask = shared.bloomMask;
   pos_t* sel = bloomSel.data();
   auto* hs = bloomHashes.data();
   size_t m = 0, i = 0;
#if defined(__AVX512F__) && !defined(VW_POS_16)
   const __m512i vMask = _mm512_set1_epi64((long long)mask);
   const __m512i b6 = _mm512_set1_epi64(63);
   const __m512i one = _mm512_set1_epi64(1);
   const __m512i lane = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4,
                                         3, 2, 1, 0);
   for (; i + 8 <= n; i += 8) {
      const __m512i h = _mm512_loadu_si512(probeHashes + i);
      const __m512i words = _mm512_i64gather_epi64(
          _mm512_and_si512(_mm512_srli_epi64(h, 40), vMask),
          (const long long*)bloom, 8);
      __m512i bits = _mm512_sllv_epi64(one, _mm512_and_si512(h, b6));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 6), b6)));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 12), b6)));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 18), b6)));
      const __mmask8 pass =
          _mm512_cmpeq_epi64_mask(_mm512_and_si512(words, bits), bits);
      const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)i), lane);
      compressStore32(sel + m, pass, idx);
      compressStore64(hs + m, pass, h);
      m += __builtin_popcount(pass);
   }
#endif
   for (; i < n; ++i) {
      const auto h = probeHashes[i];
      const uint64_t b = bloomBits(h);
      if ((bloom[(h >> 40) & mask] & b) == b) sel[m] = pos_t(i), hs[m++] = h;
   }
   bloomCount = m;
   bloomOn = true;
   if (m * 2 > n) bloomSkip = 31; // mostly hits: filter costs more than it saves
}
#endif // VW_JOIN_BLOOM

#ifdef VW_JOIN_FUSED_PROBE
// VW_JOIN_FUSED_PROBE: the join hashes its probe keys itself. Today the
// probeHash expression writes all n hashes (8 B each), then the Bloom pass
// reads them all back and copies the survivors, and the join reads those
// again; with 90-99% of probes rejected (TPC-H Q3/Q5/Q9) most of that traffic
// is for hashes nobody uses. Here, for a single int32 probe key:
//   mode A (VW_JOIN_BLOOM filter active): hash 8 keys in registers, test the
//     filter, compress only the survivors (index + hash); a survivor's hash
//     is also written to its probeHashes slot for the chain-following loop;
//   mode B (no filter): VW_NEW_JOIN's first pass hashes 8 keys in registers
//     and runs the directory step on them, storing them for the followups.
// The hashes are bit-identical to the plan's hash primitive (DEFAULT_HASH
// with primitives::seed, through the same SIMD kernel when VW_CRC32_VPCLMUL /
// VW_SIMD_HASH apply); join_sel's keys are gathered through the selection.
namespace {
#ifdef VW_USE_CRC32
using FusedHash = runtime::CRC32Hash;
#else
using FusedHash = runtime::MurMurHash;
#endif
inline runtime::Hashmap::hash_t fusedHash1(int32_t k) {
   return FusedHash()(k, primitives::seed);
}
#if defined(__AVX512F__) && !defined(VW_POS_16)
/// hashes of keys[i..i+7] (Sel: keys[sel[i..i+7]])
template <bool Sel>
inline __attribute__((always_inline)) __m512i
fusedHash8(const int32_t* keys, const pos_t* sel, size_t i) {
#if defined(VW_USE_CRC32) && defined(VW_CRC32_VPCLMUL) && defined(VW_HAVE_SIMD_CRC)
   static const primitives::simd_crc::Seed s(primitives::seed);
   const __m512i k = Sel ? primitives::simd_hash::gather8<int32_t>(keys, sel + i)
                         : primitives::simd_hash::widen8<int32_t>(keys + i);
   return primitives::simd_crc::crc_hash8(k, s);
#elif !defined(VW_USE_CRC32) && defined(VW_SIMD_HASH) && defined(VW_HAVE_SIMD_HASH)
   const __m512i k = Sel ? primitives::simd_hash::gather8<int32_t>(keys, sel + i)
                         : primitives::simd_hash::widen8<int32_t>(keys + i);
   return primitives::simd_hash::murmur(
       k, _mm512_set1_epi64((long long)primitives::seed));
#else
   // scalar hashes assembled in registers (no store + vector reload)
   auto h = [&](size_t l) {
      return (long long)fusedHash1(Sel ? keys[sel[i + l]] : keys[i + l]);
   };
   __m128i a = _mm_cvtsi64_si128(h(0)), b = _mm_cvtsi64_si128(h(2));
   __m128i c = _mm_cvtsi64_si128(h(4)), d = _mm_cvtsi64_si128(h(6));
   a = _mm_insert_epi64(a, h(1), 1);
   b = _mm_insert_epi64(b, h(3), 1);
   c = _mm_insert_epi64(c, h(5), 1);
   d = _mm_insert_epi64(d, h(7), 1);
   return _mm512_inserti64x4(
       _mm512_castsi256_si512(
           _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1)),
       _mm256_inserti128_si256(_mm256_castsi128_si256(c), d, 1), 1);
#endif
}
#endif
} // namespace

#ifdef VW_JOIN_BLOOM
void Hashjoin::fusedHashFilter(size_t n) {
   if (bloomSel.size() < n) {
      bloomSel.resize(n);
      bloomHashes.resize(n);
   }
   const bool sel = fusedSel != nullptr;
   const int32_t* keys = static_cast<const int32_t*>(
       sel ? fusedSel->param2 : fusedDense->param1);
   const pos_t* ksel =
       sel ? static_cast<const pos_t*>(fusedSel->outputSelectionV) : nullptr;
   const uint64_t* bloom = shared.bloom.get();
   const uint64_t mask = shared.bloomMask;
   pos_t* out = bloomSel.data();
   auto* hs = bloomHashes.data();
   size_t m = 0, i = 0;
#if defined(__AVX512F__) && !defined(VW_POS_16)
   const __m512i vMask = _mm512_set1_epi64((long long)mask);
   const __m512i b6 = _mm512_set1_epi64(63);
   const __m512i one = _mm512_set1_epi64(1);
   const __m512i lane = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4,
                                         3, 2, 1, 0);
   for (; i + 8 <= n; i += 8) {
      const __m512i h = sel ? fusedHash8<true>(keys, ksel, i)
                            : fusedHash8<false>(keys, ksel, i);
      const __m512i words = _mm512_i64gather_epi64(
          _mm512_and_si512(_mm512_srli_epi64(h, 40), vMask),
          (const long long*)bloom, 8);
      __m512i bits = _mm512_sllv_epi64(one, _mm512_and_si512(h, b6));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 6), b6)));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 12), b6)));
      bits = _mm512_or_si512(
          bits, _mm512_sllv_epi64(one, _mm512_and_si512(_mm512_srli_epi64(h, 18), b6)));
      const __mmask8 pass =
          _mm512_cmpeq_epi64_mask(_mm512_and_si512(words, bits), bits);
      if (!pass) continue;
      compressStore32(out + m, pass,
                      _mm512_add_epi32(_mm512_set1_epi32((int)i), lane));
      compressStore64(hs + m, pass, h);
      m += __builtin_popcount(pass);
   }
#endif
   for (; i < n; ++i) {
      const auto h = fusedHash1(sel ? keys[ksel[i]] : keys[i]);
      const uint64_t b = bloomBits(h);
      if ((bloom[(h >> 40) & mask] & b) == b) out[m] = pos_t(i), hs[m++] = h;
   }
   // survivors' hashes where the chain-following loop reads them
   for (size_t k = 0; k < m; ++k) probeHashes[out[k]] = hs[k];
   bloomCount = m;
   bloomOn = true;
   if (m * 2 > n) bloomSkip = 31;
}
#endif // VW_JOIN_BLOOM
#endif // VW_JOIN_FUSED_PROBE

#ifdef VW_NEW_JOIN
// VW_NEW_JOIN: the join / join_sel refresh from andrew_pseudocode.md.
//
// NOTE: by default (VW_NEW_JOIN_FULL=tag) a probe is `full` only when the
// directory pointer's tag bit for its hash is set (Hashmap::tag: bit 48 + top
// 4 hash bits), not merely when its slot is occupied. Probes whose tag bit is
// clear cannot be in the chain and are dropped without touching an entry.
// VW_NEW_JOIN_FULL=occupied restores the slot-occupied check.
//
// Pass 1 (load keys, crc32 into h[]) is the probe hash expression the plan
// already evaluates into probeHashes before the join function runs (the
// CRC32 hash primitive; hash_sel for join_sel, which gathers key + sel[i]).
// Pass 2, 8 probes at a time:
//   slot  = h & mask                      (hash % bins)
//   head  = gather(directory[slot])       (chain head, tag bits stripped)
//   full  = tag bit of h set in the directory pointer (VW_NEW_JOIN_FULL_TAG,
//           the default: the probe may be in the chain), or
//           head != null (VW_NEW_JOIN_FULL=occupied: slot occupied)
//   match = full && gather(head->hash) == h
//   real  = compress(i, match)            -> head is a candidate match
//   maybe = compress(i, full && !match)   -> head is not, the chain may be
// The pseudocode's real = match AND !full can never hold under either
// meaning of full (a head can only match where full holds); real = match is
// the satisfiable reading. Probes without full drop out. With the tag
// reading most non-matching probes (90-99% in TPC-H Q3/Q9) drop out at the
// directory, without touching an entry; with occupied they gather a random
// head entry (Q3 3.8x, Q9 2.7x slower than joinAllParallel on Zen 4).
// Then the maybe list is appended to the real list (each probe's chain
// successor) and the concatenated list is walked through the existing
// followup loop, following hash chains as necessary. Reals that are false
// (equal hash, different key) are removed by keyEquality after the join,
// as for the other join variants.
// AVX-512 (compile time, 64-bit hashes, 32-bit pos_t) does the 8 lanes with
// gathers, compares and compress stores; otherwise an 8-lane scalar block
// with the same steps; fewer than 8 probes left go through a plain scalar
// loop.
namespace {
/// full for one probe: directory pointer dv, probe hash h
inline bool newJoinFull(uint64_t dv, runtime::Hashmap::hash_t h) {
#ifdef VW_NEW_JOIN_FULL_TAG
   // Hashmap::tag: bit (pointer bits - 16) + top 4 hash bits
   const uint64_t tag = uint64_t(1)
                        << ((h >> (sizeof(h) * 8 - 4)) + (sizeof(dv) * 8 - 16));
   return (dv & tag) != 0;
#else
   return (dv << 16) != 0; // pointer bits (tag bits stripped) non-null
#endif
}
} // namespace

template <bool Sel> size_t Hashjoin::joinNewFirstPass(pos_t& followupWrite) {
   using EH = runtime::Hashmap::EntryHeader;
   auto& ht = shared.ht;
   const auto* dir = reinterpret_cast<const uint64_t*>(ht.entries);
   const uint64_t mask = ht.mask;
   const uint64_t maskPtr = ht.maskPointer;
#ifdef VW_JOIN_FUSED_PROBE
   // mode B: this pass hashes the probe keys (and stores the hashes for the
   // followup loop) instead of reading probeHashes
   const bool fc = fusedCompute;
   const bool fsel = fusedSel != nullptr;
   const int32_t* fkeys =
       fc ? static_cast<const int32_t*>(fsel ? fusedSel->param2 : fusedDense->param1)
          : nullptr;
   const pos_t* fks =
       fc && fsel ? static_cast<const pos_t*>(fusedSel->outputSelectionV) : nullptr;
#endif
#ifdef VW_JOIN_BLOOM
   // the probe list: Bloom survivors (index + hash), or all probes
   const bool bf = bloomOn;
   const size_t n = bf ? bloomCount : cont.numProbes;
   const auto* H = bf ? bloomHashes.data() : probeHashes;
   const pos_t* I = bf ? bloomSel.data() : nullptr;
#define VW_NJ_ID(k) (I ? I[k] : pos_t(k))
#else
   const size_t n = cont.numProbes;
   const auto* H = probeHashes;
#define VW_NJ_ID(k) pos_t(k)
#endif
   if (newMaybeIds.size() < n) {
      newMaybeIds.resize(n);
      newMaybeEntries.resize(n);
   }
   pos_t* maybeIds = newMaybeIds.data();
   EH** maybeEntries = newMaybeEntries.data();
   size_t found = 0, realFollow = 0, maybes = 0;
   size_t i = 0;

#if defined(__AVX512F__) && !(defined(HASH_SIZE) && HASH_SIZE == 32) &&       \
    !defined(VW_POS_16)
   const __m512i vMask = _mm512_set1_epi64((long long)mask);
   const __m512i vMaskPtr = _mm512_set1_epi64((long long)maskPtr);
   const __m512i zero = _mm512_setzero_si512();
   const __m512i hashOff =
       _mm512_set1_epi64((long long)offsetof(EH, hash));
   static_assert(offsetof(EH, next) == 0, "next expected first in the entry");
   const __m512i lane = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4,
                                         3, 2, 1, 0);
#ifdef VW_NEW_JOIN_FULL_TAG
   // Hashmap::tag: bit 48 + (top 4 hash bits) of the directory pointer
   const __m512i tagBase = _mm512_set1_epi64(48);
   const __m512i one = _mm512_set1_epi64(1);
#endif
   for (; i + 8 <= n; i += 8) {
#ifdef VW_JOIN_FUSED_PROBE
      __m512i h;
      if (fc) {
         h = fsel ? fusedHash8<true>(fkeys, fks, i) : fusedHash8<false>(fkeys, fks, i);
         _mm512_storeu_si512(probeHashes + i, h);
      } else
         h = _mm512_loadu_si512(H + i);
#else
      const __m512i h = _mm512_loadu_si512(H + i);
#endif
      const __m512i slot = _mm512_and_si512(h, vMask);
      const __m512i dv = _mm512_i64gather_epi64(slot, (const long long*)dir, 8);
      const __m512i head = _mm512_and_si512(dv, vMaskPtr);
#ifdef VW_NEW_JOIN_FULL_TAG
      const __m512i tag = _mm512_sllv_epi64(
          one, _mm512_add_epi64(_mm512_srli_epi64(h, 60), tagBase));
      const __mmask8 full = _mm512_test_epi64_mask(dv, tag);
#else
      const __mmask8 full = _mm512_cmpneq_epi64_mask(head, zero);
#endif
      // No lane can be in its chain (most blocks of a mostly-missing probe
      // stream without VW_JOIN_BLOOM): skip the two dependent gathers (head
      // hash, next) and the compress stores. A masked gather with an empty
      // mask still issues.
      if (!full) continue;
      const __m512i headHash = _mm512_mask_i64gather_epi64(
          zero, full, _mm512_add_epi64(head, hashOff), nullptr, 1);
      const __mmask8 match = _mm512_mask_cmpeq_epi64_mask(full, headHash, h);
      const __mmask8 real = match;
      const __mmask8 maybe = full & (__mmask8)~match;
      const __m512i next =
          _mm512_mask_i64gather_epi64(zero, full, head, nullptr, 1);
      const __mmask8 hasNext = _mm512_mask_cmpneq_epi64_mask(full, next, zero);
      // probe index (followups), probe id (output)
#ifdef VW_JOIN_BLOOM
      const __m512i idx =
          I ? _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(I + i)))
            : _mm512_add_epi32(_mm512_set1_epi32((int)i), lane);
      const __m512i out =
          !Sel ? idx
          : I  ? _mm512_castsi256_si512(_mm256_i32gather_epi32(
                    (const int*)probeSel, _mm512_castsi512_si256(idx), 4))
               : _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i*)(probeSel + i)));
#else
      const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)i), lane);
      const __m512i out =
          Sel ? _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i*)(probeSel + i)))
              : idx;
#endif
      // real list: the matched heads
      compressStore64(buildMatches + found, real, head);
      compressStore32(probeMatches + found, real, out);
      found += __builtin_popcount(real);
      // successors of the real list, then of the maybe list (appended below)
      const __mmask8 rf = real & hasNext, mf = maybe & hasNext;
      compressStore64(followupEntries + realFollow, rf, next);
      compressStore32(followupIds + realFollow, rf, idx);
      realFollow += __builtin_popcount(rf);
      compressStore64(maybeEntries + maybes, mf, next);
      compressStore32(maybeIds + maybes, mf, idx);
      maybes += __builtin_popcount(mf);
   }
#else
   for (; i + 8 <= n; i += 8) {
      EH* head[8];
      EH* next[8];
      bool full[8], match[8];
      for (size_t l = 0; l < 8; ++l) {
#ifdef VW_JOIN_FUSED_PROBE
         const auto h = fc ? (probeHashes[i + l] = fusedHash1(
                                  fsel ? fkeys[fks[i + l]] : fkeys[i + l]))
                           : H[i + l];
#else
         const auto h = H[i + l];
#endif
         const uint64_t dv = dir[h & mask];
         head[l] = reinterpret_cast<EH*>(dv & maskPtr);
         full[l] = newJoinFull(dv, h);
         match[l] = full[l] && head[l]->hash == h;
         next[l] = full[l] ? head[l]->next : nullptr;
      }
      for (size_t l = 0; l < 8; ++l) // real list
         if (match[l]) {
            const pos_t id = VW_NJ_ID(i + l);
            buildMatches[found] = head[l];
            probeMatches[found++] = Sel ? probeSel[id] : id;
         }
      for (size_t l = 0; l < 8; ++l) {
         if (!next[l]) continue;
         if (match[l]) {
            followupIds[realFollow] = VW_NJ_ID(i + l);
            followupEntries[realFollow++] = next[l];
         } else { // full && !match: maybe list
            maybeIds[maybes] = VW_NJ_ID(i + l);
            maybeEntries[maybes++] = next[l];
         }
      }
   }
#endif
   // fewer than 8 probes left
   for (; i < n; ++i) {
#ifdef VW_JOIN_FUSED_PROBE
      const auto h = fc ? (probeHashes[i] = fusedHash1(fsel ? fkeys[fks[i]] : fkeys[i]))
                        : H[i];
#else
      const auto h = H[i];
#endif
      const pos_t id = VW_NJ_ID(i);
      const uint64_t dv = dir[h & mask];
      auto head = reinterpret_cast<EH*>(dv & maskPtr);
      if (!newJoinFull(dv, h)) continue;
      const bool match = head->hash == h;
      if (match) {
         buildMatches[found] = head;
         probeMatches[found++] = Sel ? probeSel[id] : id;
      }
      if (head->next) {
         if (match) {
            followupIds[realFollow] = id;
            followupEntries[realFollow++] = head->next;
         } else {
            maybeIds[maybes] = id;
            maybeEntries[maybes++] = head->next;
         }
      }
   }
#undef VW_NJ_ID
   // append the maybe list to the real list
   std::memcpy(followupIds + realFollow, maybeIds, maybes * sizeof(pos_t));
   std::memcpy(followupEntries + realFollow, maybeEntries,
               maybes * sizeof(EH*));
   followupWrite = pos_t(realFollow + maybes);
   return found;
}

template <bool Sel> pos_t Hashjoin::joinNew() {
   size_t found = 0;
   auto followup = contCon.followup;
   auto followupWrite = contCon.followupWrite;

   if (followup == followupWrite) found = joinNewFirstPass<Sel>(followupWrite);

   followupWrite %= followupBufferSize;

   // the concatenated real + maybe list: follow the hash chains (as
   // joinAllParallel / joinSelParallel, including the continuation when the
   // output buffers fill)
   while (followup != followupWrite) {
      auto remainingSpace = batchSize - found;
      auto nrFollowups = followup <= followupWrite
                             ? followupWrite - followup
                             : followupBufferSize - (followup - followupWrite);
      auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
      for (size_t j = 0; j < fittingElements; ++j) {
         size_t i = followupIds[followup];
         auto entry = followupEntries[followup];
         followup = (followup + 1) % followupBufferSize;
         auto hash = probeHashes[i];
         if (entry->hash == hash) {
            buildMatches[found] = entry;
            probeMatches[found++] = Sel ? probeSel[i] : pos_t(i);
         }
         if (entry->next != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry->next;
            followupWrite = (followupWrite + 1) % followupBufferSize;
         }
      }
      if (fittingElements < nrFollowups) {
         // continuation
         contCon.followupWrite = followupWrite;
         contCon.followup = followup;
         return found;
      }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followup = 0;
   contCon.followupWrite = 0;
   return found;
}

pos_t Hashjoin::joinAllNew() { return joinNew<false>(); }
pos_t Hashjoin::joinSelNew() { return joinNew<true>(); }
#endif // VW_NEW_JOIN

template <typename T, typename HT>
void INTERPRET_SEPARATE insertAllEntries(T& allocations, HT& ht,
                                         size_t ht_entry_size) {
   for (auto& block : allocations) {
      auto start =
          reinterpret_cast<runtime::Hashmap::EntryHeader*>(block.first);
      ht.insertAll_tagged(start, block.second, ht_entry_size);
   }
}

pos_t Hashjoin::joinBoncz() {
   size_t followupWrite = contCon.followupWrite;
   size_t found = 0;
   if (followupWrite == 0)
      for (size_t i = 0, end = cont.numProbes; i < end; ++i) {
         auto hash = probeHashes[i];
         auto entry = shared.ht.find_chain_tagged(hash);
         if (entry != shared.ht.end()) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite] = entry;
            followupWrite += 1;
         }
      }

   while (followupWrite > 0) {
      size_t e = followupWrite;
      followupWrite = 0;
      for (size_t j = 0; j < e; j++) {
         auto i = followupIds[j];
         auto entry = followupEntries[j];
         // auto hash = probeHashes[i];
         // if (entry->hash == hash) {
         buildMatches[found] = entry;
         probeMatches[found++] = i;
         // }
         if (entry->next) {
            followupIds[followupWrite] = i;
            followupEntries[followupWrite++] = entry->next;
         }
      }
      if (followupWrite == 0) {
         cont.nextProbe = cont.numProbes;
         contCon.followupWrite = followupWrite;
         return found;
      } else if(found + followupWrite >= batchSize){
         contCon.followupWrite = followupWrite;
         assert(found);
         return found;
     }
   }
   cont.nextProbe = cont.numProbes;
   contCon.followupWrite = followupWrite;
   return 0;
}

#ifdef VW_JOIN_DISPATCH
// VW_JOIN_DISPATCH: without it, next() re-decides per probe vector whether
// the probe hash is fused (fusedReady), whether the Bloom filter runs (its
// adaptive skip counter) and whether the join is VW_NEW_JOIN (pointer
// compares). Everything those depend on is fixed once the build is done:
//   plan facts       semi marker, key count/types (semiCandidate,
//                    fusedReady), build keys through a selection
//                    (buildKeysSelected), the join function
//   build summary    build rows (filter allocated iff >= MIN_KEYS on a
//                    filtered build side), key range (semi bitmap size)
// so resolvePaths() picks the per-vector probe step once:
//   fused + filter   stepFusedBloom  hash in registers, filter, survivors
//   fused + new join stepFusedNew    joinNewFirstPass hashes (mode B)
//   filter           stepHashBloom   probe hash expression, then filter
//   neither          stepHash        probe hash expression
// The filter stays on for the whole operator: no data-dependent switch.
// semiActive (the bitmap path) is decided by the build, as without dispatch.
void Hashjoin::resolvePaths() {
   bloomOn = false;
   fusedCompute = false;
   const bool bloom = shared.bloom != nullptr;
   const bool fused = fusedReady();
#ifdef VW_NEW_JOIN
   const bool newJoin = join == &Hashjoin::joinAllNew || join == &Hashjoin::joinSelNew;
#else
   const bool newJoin = false;
#endif
   if (fused && bloom) {
      probeStep = &Hashjoin::stepFusedBloom;
   } else if (fused && newJoin) {
      probeStep = &Hashjoin::stepFusedNew;
      fusedCompute = true;
   } else if (bloom) {
      probeStep = &Hashjoin::stepHashBloom;
   } else {
      probeStep = &Hashjoin::stepHash;
   }
   // VW_DISPATCH_TRACE=1: one line per join (worker 0) with what was chosen
   static const bool trace = std::getenv("VW_DISPATCH_TRACE") != nullptr;
   if (trace && runtime::this_worker->worker_id == 0)
      std::fprintf(stderr,
                   "vw join %p: build=%zu dirKiB=%zu L1KiB=%zu keysSelected=%d "
                   "semi=%d semiBitmap=%d bloom=%d fused=%d newJoin=%d step=%s\n",
                   static_cast<void*>(this), size_t(shared.found.load()),
                   size_t(shared.ht.mask + 1) * sizeof(void*) / 1024,
                   runtime::cacheBytes(1) / 1024,
                   int(buildKeysSelected), int(semiJoin), int(semiActive),
                   int(bloom), int(fused), int(newJoin),
                   probeStep == &Hashjoin::stepFusedBloom  ? "fused+bloom"
                   : probeStep == &Hashjoin::stepFusedNew  ? "fused+newjoin"
                   : probeStep == &Hashjoin::stepHashBloom ? "hash+bloom"
                                                           : "hash");
}
void Hashjoin::stepHash(size_t n) { probeHash.evaluate(n); }
void Hashjoin::stepHashBloom(size_t n) {
   probeHash.evaluate(n);
   bloomFilter(n);
}
void Hashjoin::stepFusedBloom(size_t n) { fusedHashFilter(n); }
void Hashjoin::stepFusedNew(size_t) {}
#endif

size_t Hashjoin::next() {
   using runtime::Hashmap;
   // --- build
   if (!consumed) {
      size_t found = 0;
      // --- build phase 1: materialize ht entries
      for (auto n = left->next(); n != EndOfStream; n = left->next()) {
         found += n;
         // build hashes
         buildHash.evaluate(n);
         // scatter hash, keys and values into ht entries
         auto alloc =
             runtime::this_worker->allocator.allocate(n * ht_entry_size);
         if (!alloc) throw std::runtime_error("malloc failed");
         allocations.push_back(std::make_pair(alloc, n));
         scatterStart = reinterpret_cast<decltype(scatterStart)>(alloc);
         buildScatter.evaluate(n);
      }

      // --- build phase 2: insert ht entries
      shared.found.fetch_add(found);
#ifdef VW_JOIN_BLOOM
      auto allocBloom = [&](size_t keys) {
         size_t words = 1;
         while (words * 64 < keys * VW_JOIN_BLOOM_BITS) words <<= 1;
         shared.bloom.reset(new uint64_t[words]());
         shared.bloomMask = words - 1;
      };
#endif
#ifdef VW_JOIN_DISPATCH
      // plan fact: only a filtered build side (build keys read through a
      // selection) leaves probes that miss; an unfiltered build side's
      // foreign-key probes all hit, where the filter only costs. A semi
      // candidate gets its filter only if the bitmap is ruled out (below).
      const bool wantBloom = buildKeysSelected;
      // hardware fact: the filter pays once the directory (8 B per slot) is
      // larger than L1. A probe pipeline streams its other columns through
      // L2, so only an L1-sized directory stays resident (run_joindispatchbench
      // bloom_size: in isolation the crossover is at directory == L2; with a
      // 32 B/row competing stream it moves to ~L2/4; end to end, Q9 J4's
      // 512 KiB directory needs the filter). Unknown L1: VW_JOIN_BLOOM_MIN_KEYS.
      auto bloomPays = [&]() {
         const size_t l1 = runtime::cacheBytes(1);
         const size_t dirBytes = size_t(shared.ht.mask + 1) * sizeof(void*);
         return l1 ? dirBytes >= l1 : shared.found.load() >= VW_JOIN_BLOOM_MIN_KEYS;
      };
#endif
      barrier([&]() {
         auto globalFound = shared.found.load();
         if (globalFound) shared.ht.setSize(globalFound);
#ifdef VW_JOIN_BLOOM
         shared.bloom.reset();
#ifdef VW_JOIN_DISPATCH
         if (globalFound && wantBloom && !semiCandidate() && bloomPays())
            allocBloom(globalFound);
#else
         if (globalFound >= VW_JOIN_BLOOM_MIN_KEYS) allocBloom(globalFound);
#endif
#endif
      });
      auto globalFound = shared.found.load();
      if (globalFound == 0) {
         consumed = true;
         return EndOfStream;
      }
      insertAllEntries(allocations, shared.ht, ht_entry_size);
#ifdef VW_JOIN_BLOOM
      if (shared.bloom) bloomInsert();
#endif
#ifdef VW_JOIN_SEMI
      const bool semiCand = semiCandidate();
      if (semiCand) semiRange();
#endif
      consumed = true;
      barrier(); // wait for all threads to finish build phase
#ifdef VW_JOIN_SEMI
      if (semiCand) {
         barrier([&]() {
            shared.semiOn = false;
            const int64_t lo = shared.semiMin.load(), hi = shared.semiMax.load();
#ifdef VW_JOIN_DISPATCH
            // bitmap no larger than VW_JOIN_SEMI_MAX_BYTES (run_joindispatchbench
            // semi sweep: past that the hash path probes faster)
            if (hi >= lo && hi - lo < int64_t(VW_JOIN_SEMI_MAX_BYTES) * 8) {
#else
            if (hi >= lo && hi - lo < (int64_t(1) << 27)) {
#endif
               shared.semiBits.reset(new uint32_t[((hi - lo) >> 5) + 1]());
               shared.semiOn = true;
            }
#ifdef VW_JOIN_DISPATCH
            const size_t keys = shared.found.load();
            if (!shared.semiOn && wantBloom && bloomPays()) allocBloom(keys);
#endif
         });
         if (shared.semiOn) semiSetBits();
#ifdef VW_JOIN_DISPATCH
         else if (shared.bloom) bloomInsert();
#endif
         barrier();
         semiActive = shared.semiOn;
      }
#endif
#ifdef VW_JOIN_DISPATCH
      resolvePaths(); // once per operator; the build summary is known now
#endif
   }
   // --- lookup
#ifdef VW_JOIN_SEMI
   if (semiActive) {
      // existence only: probe positions whose key is in the build bitmap
      while (true) {
         const auto n = right->next();
         if (n == EndOfStream) return EndOfStream;
         const pos_t found = semiProbe(n);
         if (found) return found;
      }
   }
#endif
   while (true) {
      if (cont.nextProbe >= cont.numProbes) {
         cont.numProbes = right->next();
         cont.nextProbe = 0;
         if (cont.numProbes == EndOfStream) return EndOfStream;
#ifdef VW_JOIN_DISPATCH
         (this->*probeStep)(cont.numProbes);
      }
#else
#ifdef VW_JOIN_FUSED_PROBE
         fusedCompute = false;
         if (fusedReady()) {
#ifdef VW_JOIN_BLOOM
            bloomOn = false;
            if (shared.bloom && !bloomSkip) {
               fusedHashFilter(cont.numProbes); // mode A
               goto probe;
            }
            if (shared.bloom) --bloomSkip;
#endif
#ifdef VW_NEW_JOIN
            if (join == &Hashjoin::joinAllNew || join == &Hashjoin::joinSelNew) {
               fusedCompute = true; // mode B: joinNewFirstPass hashes
               goto probe;
            }
#endif
         }
#endif
         probeHash.evaluate(cont.numProbes);
#ifdef VW_JOIN_BLOOM
         bloomOn = false;
         if (shared.bloom) {
            if (bloomSkip)
               --bloomSkip;
            else
               bloomFilter(cont.numProbes);
         }
#endif
#ifdef VW_JOIN_FUSED_PROBE
      probe:;
#endif
      }
#endif // VW_JOIN_DISPATCH
      // create join pair vectors with matching hashes (Entry*, pos), where
      // Entry* is for the build side, pos a selection index to the right side
      auto n = (this->*join)();
      // check key equality and remove non equal keys from join result
      n = keyEquality.evaluate(n);
      if (n == 0) continue;
      // materialize build side
      buildGather.evaluate(n);
      return n;
   }
}

Hashjoin::Hashjoin(Shared& sm) : shared(sm) {}

Hashjoin::~Hashjoin() {
   // for (auto& block : allocations) free(block.first);
}

HashGroup::HashGroup(Shared& s)
    : shared(s), preAggregation(*this), globalAggregation(*this) {
   maxFill = ht.setSize(initialMapSize);
}
HashGroup::~HashGroup() {
   // for (auto& alloc : preAggregation.allocations) free(alloc.first);
   // for (auto& alloc : globalAggregation.allocations) free(alloc.first);
}

pos_t HashGroup::findGroupsFromPartition(void* data, size_t n) {
   globalAggregation.groupHashes = reinterpret_cast<hash_t*>(data);
   return globalAggregation.findGroups(n, ht);
}

size_t HashGroup::next() {
   using header_t = decltype(ht)::EntryHeader;
   if (!cont.consumed) {
      /// ------ phase 1: local preaggregation
      auto& spill = shared.spillStorage.local();
      auto entry_size = preAggregation.ht_entry_size;

      auto flushAndClear = [&]() INTERPRET_SEPARATE {
#ifdef VW_GROUP_AGGR
         groups.clear();
#endif
         assert(offsetof(header_t, next) + sizeof(header_t::next) ==
                offsetof(header_t, hash));
         for (auto& alloc : preAggregation.allocations) {
            for (auto entry = reinterpret_cast<header_t*>(alloc.first),
                      end = addBytes(entry, alloc.second * entry_size);
                 entry < end; entry = addBytes(entry, entry_size))
               spill.push_back(&entry->hash, entry->hash);
         }
         preAggregation.allocations.clear();
         preAggregation.clearHashtable(ht);
#ifdef VW_GROUP_BATCH_CREATE
         newBlock = nullptr, newUsed = newCap = 0;
#endif
      };

      if (packedKeys.size() < vecSize * totalKeySize) {
         packedKeys.resize(vecSize * totalKeySize);
      }
#ifdef VW_GROUP_AGGR
      if (groups.capacity() < vecSize) {
         groups.reserve(vecSize);
      }
#endif

#ifdef VW_GROUP_DISPATCH
      if (!pathsResolved) resolvePaths();
      for (pos_t n = child->next(); n != EndOfStream; n = child->next()) {
         (this->*keyStep)(n);
         (this->*aggStep)(n);
         if (preAggregation.entries_in_ht >= maxFill) flushAndClear();
      }
      if (false)
#endif
      for (pos_t n = child->next(); n != EndOfStream; n = child->next()) {
#ifdef VW_GROUP_AGGR
         for (auto entry : groups) {
            entry->group->size = 0;
         }
#endif

#ifdef VW_GROUP_NO_CONCAT
         // One dense key of 1/2/4/8 bytes (TPC-H Q18: l_orderkey) already is
         // its packed form; Concat would only copy it.
         if (keyColumns.size() == 1 && !keyColumns.front().sel &&
             (totalKeySize == 1 || totalKeySize == 2 || totalKeySize == 4 ||
              totalKeySize == 8)) {
            keyData = static_cast<char*>(keyColumns.front().data);
         } else
         {
            Concat(n);
            keyData = packedKeys.data();
         }
#else
         Concat(n);
#endif
#ifndef VW_FUSE_HASH
         Hash(n);
#endif
         Lookup(n);

#ifdef VW_AGGR_FUSED
         if (fusable && fusedAggrs.size() > 1)
            updateGroupsFused(n);
         else
#endif
         updateGroups.evaluate(n);
         if (preAggregation.entries_in_ht >= maxFill) flushAndClear();
      }
      flushAndClear();
      barrier();

      cont.consumed = true;
      cont.partition = shared.partition.fetch_add(1);
      cont.partitionNeedsAggregation = true;
   }

   /// ------ phase 2: global aggregation
   for (; cont.partition < nrPartitions;) {
      if (cont.partitionNeedsAggregation) {
         auto partNr = cont.partition;
         for (auto& threadPartitions : shared.spillStorage.threadData) {
            auto& partition = threadPartitions.second.getPartitions()[partNr];
            for (auto chunk = partition.first; chunk; chunk = chunk->next) {
               auto elementSize = threadPartitions.second.entrySize;
               auto nPart = partition.size(chunk, elementSize);
               for (size_t n = std::min(nPart, vecSize), pos = 0; n;
                    nPart -= n, pos += n, n = std::min(nPart, vecSize)) {
                  auto data = addBytes(chunk->data<void>(), pos * elementSize);
                  globalAggregation.rowData = data;
#ifdef VW_GROUP_GLOBAL_DIRECT
                  if (globalDirectOk) {
                     globalFindOrCreate(data, n);
                     updateGroupsFromPartition.evaluate(n);
                     continue;
                  }
#endif
                  findGroupsFromPartition(data, n);
                  auto cGroups = [&]() INTERPRET_SEPARATE {
                     globalAggregation.createMissingGroups(ht, true);
                  };
                  cGroups();
                  updateGroupsFromPartition.evaluate(n);
               }
            }
         }
         cont.partitionNeedsAggregation = false;
         cont.iter = globalAggregation.allocations.begin();
      }
      if (cont.iter != globalAggregation.allocations.end()) {
         auto& block = *cont.iter;
         *globalAggregation.htMatches =
             reinterpret_cast<header_t*>(block.first);
         auto n = block.second;
#ifdef VW_GROUP_HAVING
         if (havingCondition) {
            // the condition's input for every group, then only passing groups
            cont.iter++;
            havingInput->run(n);
            const pos_t m = havingCondition->evaluate(n);
            if (m == 0) continue;
            gatherGroups.evaluate(m);
            // compact the condition's input in place (sel is ascending)
            auto compact = [&](auto* col) {
               for (pos_t j = 0; j < m; ++j) col[j] = col[havingSel[j]];
            };
            switch (havingInputSize) {
            case 4: compact(static_cast<uint32_t*>(havingInputTarget)); break;
            case 8: compact(static_cast<uint64_t*>(havingInputTarget)); break;
            default: {
               char* in = static_cast<char*>(havingInputTarget);
               for (pos_t j = 0; j < m; ++j)
                  std::memmove(in + size_t(j) * havingInputSize,
                               in + size_t(havingSel[j]) * havingInputSize,
                               havingInputSize);
            }
            }
            return m;
         }
#endif
         gatherGroups.evaluate(n);
         cont.iter++;
         return n;
      } else {
         auto htClear = [&]() INTERPRET_SEPARATE {
            globalAggregation.clearHashtable(ht);
         };
         htClear();
         cont.partitionNeedsAggregation = true;
         cont.partition = shared.partition.fetch_add(1);
      }
   }
   return EndOfStream;
}

#ifdef VW_AGGR_FUSED
// Today every aggregate is its own pass over the vector, a read-modify-write
// through entries[i] per row. With few groups (TPC-H Q1: 4) consecutive rows
// hit the same entry, so each pass is one serial chain of store-to-load
// forwards (about 4 cycles per row on Zen 4, 90% of aggr_col's samples on
// the RMW). Updating all K aggregates of the row's entry in one pass gives K
// independent chains per row (bench_aggr: 3.97 -> 1.65 ns/row for Q1's 5).
// Dense columns read through an identity selection and COUNT(*) sums a
// column of ones, so every aggregate is col[sel[i]].
namespace {
template <size_t K>
void aggrFusedBlock(pos_t n, runtime::Hashmap::EntryHeader** RES entries,
                    const int64_t* const* cols, const pos_t* const* sels,
                    const size_t* offsets) {
   const int64_t* c[K];
   const pos_t* s[K];
   size_t o[K];
   for (size_t d = 0; d < K; ++d) c[d] = cols[d], s[d] = sels[d], o[d] = offsets[d];
   for (pos_t i = 0; i < n; i++) {
      char* e = reinterpret_cast<char*>(entries[i]);
      for (size_t d = 0; d < K; ++d)
         *reinterpret_cast<int64_t*>(e + o[d]) += c[d][s[d][i]];
   }
}
} // namespace

void HashGroup::updateGroupsFused(pos_t n) {
   if (identitySel.size() < n) {
      identitySel.resize(std::max<size_t>(n, vecSize));
      for (size_t i = 0; i < identitySel.size(); ++i) identitySel[i] = pos_t(i);
      ones.assign(identitySel.size(), 1);
   }
   constexpr size_t kMax = 8;
   const int64_t* cols[kMax];
   const pos_t* sels[kMax];
   size_t offsets[kMax];
   auto entries = preAggregation.htMatches;
   for (size_t first = 0; first < fusedAggrs.size(); first += kMax) {
      const size_t k = std::min(kMax, fusedAggrs.size() - first);
      for (size_t d = 0; d < k; ++d) {
         auto& fa = fusedAggrs[first + d];
         offsets[d] = fa.offset;
         switch (fa.kind) {
         case FusedAggr::Col:
            cols[d] = static_cast<const int64_t*>(
                static_cast<FAggrOp*>(fa.op)->get<1>());
            sels[d] = identitySel.data();
            break;
         case FusedAggr::SelCol:
            sels[d] = static_cast<FAggrSelOp*>(fa.op)->get<1>();
            cols[d] = static_cast<const int64_t*>(
                static_cast<FAggrSelOp*>(fa.op)->get<2>());
            break;
         case FusedAggr::Count:
            cols[d] = ones.data();
            sels[d] = identitySel.data();
            break;
         }
      }
      switch (k) {
      case 1: aggrFusedBlock<1>(n, entries, cols, sels, offsets); break;
      case 2: aggrFusedBlock<2>(n, entries, cols, sels, offsets); break;
      case 3: aggrFusedBlock<3>(n, entries, cols, sels, offsets); break;
      case 4: aggrFusedBlock<4>(n, entries, cols, sels, offsets); break;
      case 5: aggrFusedBlock<5>(n, entries, cols, sels, offsets); break;
      case 6: aggrFusedBlock<6>(n, entries, cols, sels, offsets); break;
      case 7: aggrFusedBlock<7>(n, entries, cols, sels, offsets); break;
      default: aggrFusedBlock<8>(n, entries, cols, sels, offsets); break;
      }
   }
}
#endif

#ifdef VW_GROUP_GLOBAL_DIRECT
// Global phase today: findGroups (directory walk, then keyEquality with one
// keys_not_equal_row pass per key, htFollow for chains), then
// createMissingGroups, which dedups the vector's misses with a second hash
// table (partition_by_key_row per key: two chained lookups per row) before
// scattering and inserting them. After a complete pre-aggregation (TPC-H Q18:
// lineitem sorted by l_orderkey) almost every spilled row is a new, distinct
// group, so the dedup table does all that work for nothing.
// Here, as in the pre-aggregation lookup: walk the chain comparing the
// spilled row's packed keys with memcmp (row = entry without `next`), create
// a miss at once so later rows of the vector find it, and initialize all new
// entries with one buildScatter call. updateGroupsFromPartition then adds
// every row, as before. The directory is tagged (as the join's): a new group,
// the common case, is rejected from the directory slot without walking its
// chain through entries spread over the table (TPC-H Q18: ~190k groups per
// partition, 4 MB directory + 6 MB entries), and the vector's slots are
// prefetched first. Hashmap takes the tag from the hash's top 4 bits, which
// the spill partitioning (hash >> shift) makes (nearly) constant inside a
// partition, so the table gets the hash rotated right by 16: tag from bits
// 12..15, slot from bits 16.., both clear of the partition bits. Entries keep
// the original hash.
namespace {
inline HashGroup::hash_t globalKey(HashGroup::hash_t h) {
   return (h >> 16) | (h << (sizeof(h) * 8 - 16));
}
} // namespace
void HashGroup::globalFindOrCreate(void* data, size_t n) {
   auto& g = globalAggregation;
   const size_t rowSize = g.rowSize;
   const size_t entrySize = g.ht_entry_size;
   const uint32_t keySize = totalKeySize;
   constexpr size_t rowKeyOffset =
       sizeof(EntryHeader) - sizeof(EntryHeader::next);
   EntryHeader* block = nullptr;
   pos_t created = 0;
   for (size_t i = 0; i < n; ++i) {
      hash_t hash;
      std::memcpy(&hash, static_cast<const char*>(data) + i * rowSize,
                  sizeof(hash));
      ht.prefetch_slot(globalKey(hash));
   }
   for (size_t i = 0; i < n; ++i) {
      const char* row = static_cast<const char*>(data) + i * rowSize;
      hash_t hash;
      std::memcpy(&hash, row, sizeof(hash));
      EntryHeader* entry = ht.find_chain_tagged(globalKey(hash));
      for (; entry != nullptr; entry = entry->next)
         if (entry->hash == hash &&
             std::memcmp(reinterpret_cast<char*>(entry + 1), row + rowKeyOffset,
                         keySize) == 0)
            break;
      if (!entry) {
         if (!block) {
            auto alloc = groupStore.allocate(n * entrySize);
            if (!alloc) throw std::runtime_error("malloc failed");
            block = reinterpret_cast<EntryHeader*>(alloc);
         }
         entry = addBytes(block, created * entrySize);
         entry->hash = hash;
         std::memcpy(reinterpret_cast<char*>(entry + 1), row + rowKeyOffset,
                     keySize);
         ht.insert_tagged<false>(entry, globalKey(hash));
         g.groupRepresentatives[created++] = i;
      }
      g.htMatches[i] = entry;
   }
   if (!created) return;
   g.allocations.push_back(std::make_pair(block, size_t(created)));
   g.entries_in_ht += created;
   g.scatterStart = block;
   g.buildScatter.evaluate(created);
   if (g.entries_in_ht > maxFill) {
      // as createMissingGroups(allowResize): grow and reinsert all entries
      maxFill = ht.setSize(g.entries_in_ht * 2);
      for (auto& b : g.allocations)
         for (size_t k = 0; k < b.second; ++k) {
            auto e = addBytes(reinterpret_cast<EntryHeader*>(b.first),
                              k * entrySize);
            ht.insert_tagged<false>(e, globalKey(e->hash));
         }
   }
}
#endif

#ifdef VW_GROUP_NO_CONCAT
#define VW_GROUP_KEY_DATA keyData
#else
#define VW_GROUP_KEY_DATA packedKeys.data()
#endif

#ifdef VW_GROUP_DISPATCH
// VW_GROUP_DISPATCH: HashGroup's per-vector pre-aggregation used to switch on
// totalKeySize three times per vector (Concat, Hash, Lookup) and to pack
// keys even when there is one key column. What it switches on is fixed by
// the plan for the operator's lifetime, so resolvePaths() picks, once:
//   key step: one specialized loop per (key width, key mode)
//     kSingle    one key column, no selection: the column is the packed key
//     kSingleSel one key column read through its selection vector
//     kPacked    compound keys, packed by per-column copy steps resolved once
//   the loop hashes each key itself, walks the chain, and creates missing
//   groups in one consecutive block with one buildScatter per vector
//   (batched creation); with VW_GROUP_RUN_HEADS only the first row of each
//   run of equal keys is looked up;
//   aggregate step: all int64 SUM/COUNT in one pass (fused) when possible,
//     else the aggregate expression list;
//   global phase: one-pass find-or-create for memcmp-comparable keys;
//   spill: inline word copy of rows.
void HashGroup::resolvePaths() {
   pathsResolved = true;
   const bool single = keyColumns.size() == 1;
   const KeyMode mode = !single ? kPacked
                        : keyColumns.front().sel ? kSingleSel
                                                 : kSingle;
   // compound keys: each column's copy into packedKeys, resolved once
   concatSteps.clear();
   if (mode == kPacked)
      for (auto& col : keyColumns)
         concatSteps.push_back(col.size == 1   ? &HashGroup::Concat_T<uint8_t>
                               : col.size == 2 ? &HashGroup::Concat_T<uint16_t>
                               : col.size == 4 ? &HashGroup::Concat_T<uint32_t>
                               : col.size == 8 ? &HashGroup::Concat_T<uint64_t>
                                               : &HashGroup::Concat_T<char*>);
#ifdef VW_GROUP_RUN_HEADS
   constexpr bool runs = true;
   runHeads.resize(vecSize);
   runFlags.resize(vecSize);
   runEntries.resize(vecSize);
#else
   constexpr bool runs = false;
#endif
#define VW_KEY_STEP(T)                                                         \
   (mode == kSingle      ? &HashGroup::keyStepT<T, kSingle, runs>              \
    : mode == kSingleSel ? &HashGroup::keyStepT<T, kSingleSel, runs>           \
                         : &HashGroup::keyStepT<T, kPacked, runs>)
   switch (totalKeySize) {
   case 1: keyStep = VW_KEY_STEP(uint8_t); break;
   case 2: keyStep = VW_KEY_STEP(uint16_t); break;
   case 4: keyStep = VW_KEY_STEP(uint32_t); break;
   case 8: keyStep = VW_KEY_STEP(uint64_t); break;
   default: keyStep = VW_KEY_STEP(char*); break;
   }
#undef VW_KEY_STEP
   aggStep = (fusable && fusedAggrs.size() > 1) ? &HashGroup::updateGroupsFused
                                                : &HashGroup::aggEvaluate;
}

void HashGroup::concatResolved(pos_t n) {
   for (size_t c = 0; c < keyColumns.size(); ++c)
      (this->*concatSteps[c])(n, keyColumns[c]);
}

#if defined(VW_GROUP_RUN_HEADS) && defined(__AVX512BW__) &&                   \
    defined(__AVX512VL__) && !defined(VW_POS_16)
namespace {
/// lanes of 16 consecutive keys at cur that differ from the 16 at prv
/// (= cur - one key)
template <typename T>
inline __mmask16 runHeadMask16(const char* cur, const char* prv) {
   if constexpr (sizeof(T) == 1) {
      return _mm_cmpneq_epi8_mask(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(cur)),
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(prv)));
   } else if constexpr (sizeof(T) == 2) {
      return _mm256_cmpneq_epi16_mask(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cur)),
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prv)));
   } else if constexpr (sizeof(T) == 4) {
      return _mm512_cmpneq_epi32_mask(_mm512_loadu_si512(cur),
                                      _mm512_loadu_si512(prv));
   } else {
      const __mmask8 lo = _mm512_cmpneq_epi64_mask(_mm512_loadu_si512(cur),
                                                   _mm512_loadu_si512(prv));
      const __mmask8 hi = _mm512_cmpneq_epi64_mask(
          _mm512_loadu_si512(cur + 64), _mm512_loadu_si512(prv + 64));
      return __mmask16(lo | (unsigned(hi) << 8));
   }
}
} // namespace
#define VW_RUN_HEADS_SIMD 1
#endif

// VW_GROUP_RUN_HEADS (Runs): rows that repeat the previous row's key reuse
// its entry, without a data-dependent branch or an adaptive switch (it
// replaces VW_GROUP_LAST_MATCH, which branched per row and switched itself
// on at >= 70% repeats per vector, a threshold fitted between TPC-H Q1 and
// Q18). Sorted or clustered keys (Q18: lineitem by l_orderkey) and Q1's
// 64% repeats both gain. Pass 1 writes the run heads, the rows whose key differs
// from the previous row's (AVX-512: 16 keys against the 16 before them, the
// head positions register-compressed; else a branch-free scalar loop). Pass 2
// is the lookup / create below, for heads only. Pass 3 gives every row its
// run's entry, branch-free. The first row of a vector is always a head.
template <typename T, int Mode, bool Runs>
void HashGroup::keyStepT(pos_t n) {
   constexpr bool wide = std::is_same_v<T, char*>;
   const uint32_t keySize = wide ? totalKeySize : sizeof(T);
   const char* __restrict__ base;
   const pos_t* __restrict__ ksel = nullptr;
   if constexpr (Mode == kPacked) {
      concatResolved(n);
      base = packedKeys.data();
   } else {
      base = static_cast<const char*>(keyColumns.front().data);
      if constexpr (Mode == kSingleSel) ksel = keyColumns.front().sel;
   }
   auto keyAt = [&](pos_t i) -> const char* {
      if constexpr (Mode == kSingleSel)
         return base + size_t(ksel[i]) * keySize;
      else
         return base + size_t(i) * keySize;
   };
   auto hashOf = [&](const char* kp) -> hash_t {
      if constexpr (wide) {
         return hashFn.hashKey(kp, keySize, 0);
      } else {
         T key;
         std::memcpy(&key, kp, sizeof(T));
         return hashFn.hashKey(key);
      }
   };
   auto sameKey = [&](const char* a, const char* b) {
      if constexpr (wide) {
         return std::memcmp(a, b, keySize) == 0;
      } else {
         T x, y;
         std::memcpy(&x, a, sizeof(T));
         std::memcpy(&y, b, sizeof(T));
         return x == y;
      }
   };
   hash_t* __restrict__ hashes = preAggregation.groupHashes;
   EntryHeader** __restrict__ matches = preAggregation.htMatches;
   const size_t entrySize = preAggregation.ht_entry_size;
   pos_t created = 0;
   EntryHeader* firstNew = nullptr;
   pos_t nLookups = n;
#ifdef VW_GROUP_RUN_HEADS
   pos_t* __restrict__ heads = runHeads.data();
   uint8_t* __restrict__ flags = runFlags.data();
   if constexpr (Runs) {
      // pass 1: run heads and head flags; heads[cnt] = i is written for every
      // row and kept only where the key changed, so the scalar loop has no
      // branch on data
      pos_t cnt = 0, i = 0;
      if (n) {
         flags[0] = 1;
         heads[cnt++] = i++;
      }
#ifdef VW_RUN_HEADS_SIMD
      if constexpr (!wide && Mode != kSingleSel) {
         const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                10, 11, 12, 13, 14, 15);
         for (; i + 16 <= n; i += 16) {
            const char* cur = base + size_t(i) * sizeof(T);
            const __mmask16 m = runHeadMask16<T>(cur, cur - sizeof(T));
            const __m512i idx =
                _mm512_add_epi32(_mm512_set1_epi32(int(i)), iota);
            const unsigned k = __builtin_popcount(m);
            _mm512_mask_storeu_epi32(heads + cnt, __mmask16((1u << k) - 1),
                                     _mm512_maskz_compress_epi32(m, idx));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(flags + i),
                             _mm_maskz_set1_epi8(m, 1));
            cnt += k;
         }
      }
#endif
      for (; i < n; ++i) {
         const bool head = !sameKey(keyAt(i), keyAt(i - 1));
         flags[i] = head;
         heads[cnt] = i;
         cnt += head;
      }
      nLookups = cnt;
   }
#endif
   // pass 2 (Runs: heads only)
   for (pos_t j = 0; j < nLookups; ++j) {
      pos_t i = j;
#ifdef VW_GROUP_RUN_HEADS
      if constexpr (Runs) i = heads[j];
#endif
      const char* kp = keyAt(i);
      EntryHeader* entry;
      const hash_t hash = hashOf(kp);
      for (entry = ht.find_chain(hash); entry; entry = entry->next)
         if (entry->hash == hash &&
             sameKey(reinterpret_cast<const char*>(entry + 1), kp))
            break;
      if (!entry) {
         if (newUsed == newCap) {
            newCap = maxFill + vecSize;
            newUsed = 0;
            auto alloc = groupStore.allocate(newCap * entrySize);
            if (!alloc) throw std::runtime_error("malloc failed");
            newBlock = reinterpret_cast<EntryHeader*>(alloc);
            preAggregation.allocations.emplace_back(alloc, 0);
         }
         entry = addBytes(newBlock, newUsed * entrySize);
         ++newUsed;
         ++preAggregation.allocations.back().second;
         if (!created) firstNew = entry;
         entry->hash = hash;
         std::memcpy(reinterpret_cast<char*>(entry + 1), kp, keySize);
         hashes[i] = hash;
         preAggregation.groupRepresentatives[created++] = i;
         ht.insert<false>(entry, hash);
         ++preAggregation.entries_in_ht;
      }
      matches[i] = entry;
#ifdef VW_GROUP_RUN_HEADS
      if constexpr (Runs) runEntries[j] = entry;
#endif
   }
#ifdef VW_GROUP_RUN_HEADS
   if constexpr (Runs) {
      // pass 3: row i belongs to run k - 1 after adding its head flag; the
      // loop-carried chain is one add, the loads do not depend on each other
      EntryHeader* const* __restrict__ runEntry = runEntries.data();
      pos_t k = 0;
      for (pos_t i = 0; i < n; ++i) {
         k += flags[i];
         matches[i] = runEntry[k - 1];
      }
   }
#endif
   if (created) {
      preAggregation.scatterStart = firstNew;
      preAggregation.buildScatter.evaluate(created);
   }
}
#endif // VW_GROUP_DISPATCH

// CONCAT
void HashGroup::Concat(pos_t n) {
   for (const auto& col : keyColumns) {
      switch (col.size) {
         case 1: Concat_T<uint8_t>(n, col); break;
         case 2: Concat_T<uint16_t>(n, col); break;
         case 4: Concat_T<uint32_t>(n, col); break;
         case 8: Concat_T<uint64_t>(n, col); break;
         default: Concat_T<char*>(n, col); break;
      }
   }
}

template <typename T> void HashGroup::Concat_T(pos_t n, const KeyColumn& col) {
   uint32_t keySize = totalKeySize;
   uint32_t colSize = std::is_same_v<T, char*> ? col.size : sizeof(T);
   char* __restrict__ src = static_cast<char*>(col.data);
   // per-column selection: dense key buffers must not go through another
   // key's selection vector
   pos_t* __restrict__ sel = col.sel;
   char* __restrict__ dest = packedKeys.data() + col.offset;

   if (sel) {
      for (pos_t i = 0; i < n; i++) {
         std::memcpy(dest + i * keySize, src + sel[i] * colSize, colSize);
      }
   } else {
      for (pos_t i = 0; i < n; i++) {
         std::memcpy(dest + i * keySize, src + i * colSize, colSize);
      }
   }
}

// HASH
void HashGroup::Hash(pos_t n) {
   switch (totalKeySize) {
      case 1: Hash_T<uint8_t>(n); break;
      case 2: Hash_T<uint16_t>(n); break;
      case 4: Hash_T<uint32_t>(n); break;
      case 8: Hash_T<uint64_t>(n); break;
      default: Hash_T<char*>(n); break;
   }
}

template <typename T> void HashGroup::Hash_T(pos_t n) {
#if defined(VW_SIMD_HASH) && defined(VW_HAVE_SIMD_HASH) && !defined(VW_USE_CRC32)
   // packed 1/2/4/8-byte keys: AVX-512 MurmurHash64A, bit-identical to
   // hashFn.hashKey(key) below
   if constexpr (!std::is_same_v<T, char*>) {
      primitives::simd_hash::hash_keys<T>(n, VW_GROUP_KEY_DATA,
                                          preAggregation.groupHashes);
      return;
   }
#endif
   uint32_t keySize = std::is_same_v<T, char*> ? totalKeySize : sizeof(T);
   char* __restrict__ keys = VW_GROUP_KEY_DATA;
   hash_t* __restrict__ hashes = preAggregation.groupHashes;

#if defined(VW_CRC32_VPCLMUL) && defined(VW_USE_CRC32) && defined(VW_HAVE_SIMD_CRC)
   // packed 1/2/4/8-byte keys: VPCLMULQDQ CRC32Hash, 8 keys per zmm,
   // bit-identical to hashFn.hashKey(key) below
   if constexpr (!std::is_same_v<T, char*>) {
      primitives::simd_crc::hash_keys<T>(n, keys, hashes);
      return;
   }
#endif
#if defined(VW_CRC32_FAST) && defined(VW_USE_CRC32)
   // wide keys: 4 keys' chunk chains interleaved, bit-identical to the loop
   if constexpr (std::is_same_v<T, char*>) {
      hashFn.hashKeys(keys, keySize, n, 0, hashes);
      return;
   }
#endif
   for (pos_t i = 0; i < n; i++) {
      if constexpr (std::is_same_v<T, char*>) {
         hashes[i] = hashFn.hashKey(keys + i * keySize, keySize, 0);
      } else {
         T key;
         std::memcpy(&key, keys + i * keySize, keySize);
         hashes[i] = hashFn.hashKey(key);
      }
   }
}

// LOOKUP
void HashGroup::Lookup(pos_t n) {
   switch (totalKeySize) {
#define VW_LOOKUP(T) Lookup_T<T>(n)
      case 1: VW_LOOKUP(uint8_t); break;
      case 2: VW_LOOKUP(uint16_t); break;
      case 4: VW_LOOKUP(uint32_t); break;
      case 8: VW_LOOKUP(uint64_t); break;
      default: VW_LOOKUP(char*); break;
#undef VW_LOOKUP
   }
}

template <typename T> void HashGroup::Lookup_T(pos_t n) {
   uint32_t keySize = std::is_same_v<T, char*> ? totalKeySize : sizeof(T);
   char* __restrict__ keys = VW_GROUP_KEY_DATA;
   hash_t* __restrict__ hashes = preAggregation.groupHashes;
#ifdef VW_GROUP_AGGR_SEL
   pos_t* __restrict__ sel = selVec;
#endif
#ifndef VW_GROUP_AGGR
   EntryHeader** __restrict__ matches = preAggregation.htMatches;
#endif

#ifdef VW_GROUP_BATCH_CREATE
   // new groups of this vector: rows in groupRepresentatives, entries
   // consecutive from firstNew
   pos_t created = 0;
   EntryHeader* firstNew = nullptr;
   const size_t entrySize = preAggregation.ht_entry_size;
#endif
   for (pos_t i = 0; i < n; i++) {
#ifdef VW_FUSE_HASH
      hash_t hash;
      if constexpr (std::is_same_v<T, char*>) {
         hash = hashFn.hashKey(keys + i * keySize, keySize, 0);
      } else {
         T key;
         std::memcpy(&key, keys + i * keySize, keySize);
         hash = hashFn.hashKey(key);
      }
#else
      hash_t hash = hashes[i];
#endif
      EntryHeader* entry = ht.find_chain(hash);
      for (; entry != nullptr; entry = entry->next) {
         if (entry->hash == hash) {
            char* entry_key = reinterpret_cast<char*>(entry + 1);
            if (std::memcmp(keys + i * keySize, entry_key, keySize) == 0) {
               goto found;
            }
         }
      }

#ifdef VW_GROUP_BATCH_CREATE
      {
         // Today each new group is its own allocation (padded to 64 B by
         // ResetableAllocator), its own allocations record and its own
         // buildScatter.evaluate(1): 4+ interpreted ops for one row (TPC-H
         // Q18: 1.5M groups). Here the entry only gets what later rows of
         // this vector need to find it (hash, packed key); the scatter
         // (hash, keys, aggregate init) runs once for all new groups after
         // the loop.
         if (newUsed == newCap) {
            newCap = maxFill + vecSize;
            newUsed = 0;
            auto alloc = groupStore.allocate(newCap * entrySize);
            if (!alloc) throw std::runtime_error("malloc failed");
            newBlock = reinterpret_cast<EntryHeader*>(alloc);
            preAggregation.allocations.emplace_back(alloc, 0);
         }
         entry = addBytes(newBlock, newUsed * entrySize);
         ++newUsed;
         ++preAggregation.allocations.back().second;
         if (!created) firstNew = entry;
         entry->hash = hash;
         std::memcpy(reinterpret_cast<char*>(entry + 1), keys + i * keySize,
                     keySize);
#ifdef VW_FUSE_HASH
         hashes[i] = hash;
#endif
         preAggregation.groupRepresentatives[created++] = i;
         ht.insert<false>(entry, hash);
         ++preAggregation.entries_in_ht;
      }
#else
      {
         auto alloc = groupStore.allocate(preAggregation.ht_entry_size);
         if (!alloc) {
            throw std::runtime_error("malloc failed");
         }

         entry = reinterpret_cast<EntryHeader*>(alloc);
         preAggregation.allocations.emplace_back(alloc, 1);

#ifdef VW_FUSE_HASH
         hashes[i] = hash;
#endif

         preAggregation.groupRepresentatives[0] = i;
         preAggregation.scatterStart = entry;
         preAggregation.buildScatter.evaluate(1);

         ht.insert<false>(entry, hash);
         ++preAggregation.entries_in_ht;

#ifdef VW_GROUP_AGGR
         groups.push_back(entry);

         alloc = groupStore.allocate(sizeof(Group));
         if (!alloc) {
            throw std::runtime_error("malloc failed");
         }

         entry->group = reinterpret_cast<Group*>(alloc);
         entry->group->size = 0;
#endif
      }
#endif // VW_GROUP_BATCH_CREATE

      found:;
#ifdef VW_GROUP_AGGR
      pos_t next = entry->group->size;
      entry->group->pos[next] = i;
#ifdef VW_GROUP_AGGR_SEL
      // selVec is only set when some key has a selection vector; group-bys on
      // dense keys only (e.g. TPC-H Q9) have none, and the row index itself
      // is the selection. sel is loop-invariant, so this check is hoisted.
      entry->group->sel[next] = sel ? sel[i] : i;
#endif
      ++entry->group->size;
#else
      matches[i] = entry;
#endif
   }
#ifdef VW_GROUP_BATCH_CREATE
   if (created) {
      preAggregation.scatterStart = firstNew;
      preAggregation.buildScatter.evaluate(created);
   }
#endif
}
} // namespace vectorwise
