#include "vectorwise/Operators.hpp"
#include "vectorwise/Primitives.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Hashmap.hpp"
#include "common/runtime/SIMD.hpp"
#include "vectorwise/SimdCrc.hpp"
#include "vectorwise/SimdHash.hpp"
#include <algorithm>
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
      for (size_t i = 0, end = cont.numProbes; i < end; ++i) {
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
      for (size_t i = 0, end = cont.numProbes; i < end; ++i) {
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
      barrier([&]() {
         auto globalFound = shared.found.load();
         if (globalFound) shared.ht.setSize(globalFound);
      });
      auto globalFound = shared.found.load();
      if (globalFound == 0) {
         consumed = true;
         return EndOfStream;
      }
      insertAllEntries(allocations, shared.ht, ht_entry_size);
      consumed = true;
      barrier(); // wait for all threads to finish build phase
   }
   // --- lookup
   while (true) {
      if (cont.nextProbe >= cont.numProbes) {
         cont.numProbes = right->next();
         cont.nextProbe = 0;
         if (cont.numProbes == EndOfStream) return EndOfStream;
         probeHash.evaluate(cont.numProbes);
      }
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
#ifdef VW_GROUP_LAST_MATCH
#define VW_LOOKUP(T) lastMatchOn ? Lookup_T<T, true>(n) : Lookup_T<T, false>(n)
#else
#define VW_LOOKUP(T) Lookup_T<T>(n)
#endif
      case 1: VW_LOOKUP(uint8_t); break;
      case 2: VW_LOOKUP(uint16_t); break;
      case 4: VW_LOOKUP(uint32_t); break;
      case 8: VW_LOOKUP(uint64_t); break;
      default: VW_LOOKUP(char*); break;
#undef VW_LOOKUP
   }
}

#ifdef VW_GROUP_LAST_MATCH
template <typename T, bool LastMatch> void HashGroup::Lookup_T(pos_t n) {
#else
template <typename T> void HashGroup::Lookup_T(pos_t n) {
#endif
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
#ifdef VW_GROUP_LAST_MATCH
   // Sorted or clustered group keys (TPC-H Q18: lineitem by l_orderkey, ~75%
   // of rows repeat the previous key) find the previous row's entry again.
   // Compare with the previous packed key first and reuse its entry: no
   // directory load, chain walk or key compare (and with VW_FUSE_HASH no
   // hash) for those rows. The extra branch mispredicts on short runs (Q1:
   // 64% repeats, +12-20% time), so each vector counts its repeats and the
   // next one uses the check only at >= 70%.
   EntryHeader* last = nullptr;
   pos_t repeats = 0;
#endif
   for (pos_t i = 0; i < n; i++) {
#ifdef VW_GROUP_LAST_MATCH
      EntryHeader* entry;
      {
         const bool repeat =
             i > 0 && std::memcmp(keys + i * keySize, keys + (i - 1) * keySize,
                                  keySize) == 0;
         if constexpr (LastMatch) {
            if (repeat) {
               ++repeats;
               entry = last;
               goto found;
            }
         } else {
            repeats += repeat;
         }
      }
      {
#endif
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
#ifdef VW_GROUP_LAST_MATCH
      entry = ht.find_chain(hash);
#else
      EntryHeader* entry = ht.find_chain(hash);
#endif
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
#ifdef VW_GROUP_LAST_MATCH
      }
#endif

      found:;
#ifdef VW_GROUP_LAST_MATCH
      last = entry;
#endif
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
#ifdef VW_GROUP_LAST_MATCH
   if (n > 1) lastMatchOn = size_t(repeats) * 10 >= size_t(n - 1) * 7;
#endif
}
} // namespace vectorwise
