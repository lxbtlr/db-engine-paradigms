#pragma once
#include "Operations.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Database.hpp"
#include "common/runtime/Hash.hpp"
#include "common/runtime/Hashmap.hpp"
#include "common/runtime/PartitionedDeque.hpp"
#include "common/runtime/Query.hpp"
#include "vectorwise/Primitives.hpp"
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace vectorwise {

const size_t EndOfStream = 0;

class Operator {
 public:
   Operator() = default;
   Operator(Operator&&) = default;
   Operator(const Operator&) = delete;
   virtual size_t next() = 0;
   virtual ~Operator() = default;
};

class UnaryOperator : public Operator {
 public:
   std::unique_ptr<Operator> child;
};

class BinaryOperator : public Operator {
 public:
   std::unique_ptr<Operator> left;
   std::unique_ptr<Operator> right;
};

class Select : public UnaryOperator {
 public:
   std::unique_ptr<Expression> condition;
   virtual size_t next() override;
};

class Project : public UnaryOperator {
 public:
   std::vector<std::unique_ptr<Expression>> expressions;
   virtual size_t next() override;
};

class FixedAggr : public UnaryOperator {
   bool consumed = false;

 public:
   Aggregates aggregates;
   virtual size_t next() override;
};

class SharedState
/// Base class for shared operator state
{
 public:
   inline virtual ~SharedState(){};
};

class SharedStateManager {
   std::mutex m;
   std::unordered_map<size_t, std::unique_ptr<SharedState>> state;
   std::atomic<size_t> oncesExecuted;
   std::mutex onceMutex;

 public:
   SharedStateManager() : oncesExecuted(0) {}
   template <typename T> T& get(size_t i) {
      std::lock_guard<std::mutex> lock(m);
      auto s = state.find(i);
      if (s == state.end()) {
         s = state.insert(std::make_pair(i, move(std::make_unique<T>()))).first;
      }
      T* res = dynamic_cast<T*>(s->second.get());
      if (!res)
         throw std::runtime_error(
             "Failed to retrieve shared state. Wrong type found.");
      return *res;
   }

   template <typename C> void once(size_t i, C callback) {
      auto onces = oncesExecuted.load();
      if (onces > i) return; // has already been executed
      {
         std::lock_guard<std::mutex> lock(onceMutex);
         if (oncesExecuted.load() > i) return; // has already been executed
         callback();
         auto current = oncesExecuted.fetch_add(1); // mark this as executed
         if (current != i)
            throw std::runtime_error(
                "Synchronization in onces failed. Found value " +
                std::to_string(current) + ", expected " + std::to_string(i));
      }
   }
};

template <typename PAYLOAD> class DebugOperator : public UnaryOperator {
 public:
   struct Shared : public SharedState {
      std::mutex state_mutex;
      PAYLOAD data;
   };

   DebugOperator(Shared& shared, std::function<void(size_t, PAYLOAD&)> step,
                 std::function<void(PAYLOAD&)> finish)
       : shared_(shared), step_(std::move(step)), finish_(std::move(finish)) {}
   virtual size_t next() override {
      auto n = child->next();
      {
         std::lock_guard<std::mutex> lock(shared_.state_mutex);
         step_(n, shared_.data);
      }
      return n;
   }

   virtual ~DebugOperator() override {
      bool leader = runtime::barrier();
      if (leader) {
         std::lock_guard<std::mutex> lock(shared_.state_mutex);
         finish_(shared_.data);
      }
   }

 private:
   Shared& shared_;
   std::function<void(size_t, PAYLOAD&)> step_;
   std::function<void(PAYLOAD&)> finish_;
};

class Scan : public Operator {
 public:
   struct Shared : public SharedState {
      std::atomic<size_t> pos;
      Shared() : pos(0){};
   };

 private:
   size_t scanChunkSize;
   Shared& shared;
   bool needsInit;
   size_t vecInChunk;
   size_t currentChunk;
   size_t lastOffset;
   size_t nrTuples;
   size_t vecSize;

   std::vector<std::pair<void**, size_t>> consumers;

 public:
   Scan(Shared& sm, size_t nrTuples, size_t vecSize);
   /// Add consumer to scan operator, typeSize is size of
   /// type pointed to by colPtr
   void addConsumer(void** colPtr, size_t typeSize);
   virtual size_t next() override;
};

class ResultWriter : public UnaryOperator {
 public:
   struct Shared : public SharedState {
      std::unique_ptr<runtime::Query> result;
      Shared() : result(std::make_unique<runtime::Query>()) {}
   } & shared;

   struct Input {
      void* data;
      size_t elementSize;
      runtime::BlockRelation::Attribute attribute;
      Input(void* d, size_t size, runtime::BlockRelation::Attribute attr);
   };
   /// Buffers that get copied to output
   std::deque<Input> inputs;
   /// ctor
   ResultWriter(Shared& s);
   /// run the operator
   virtual size_t next() override;

 private:
   runtime::BlockRelation::Block currentBlock;
};

class Hashjoin : public BinaryOperator {
 public:
   struct Shared : public SharedState {
      std::atomic<size_t> found;
      std::atomic<bool> sizeIsSet;
      runtime::Hashmap ht;
#ifdef VW_JOIN_BLOOM
      /// VW_JOIN_BLOOM: register-blocked Bloom filter over the build side
      /// (one 64-bit word per key's block, 4 bits per key), owned by this
      /// join; nullptr for builds under VW_JOIN_BLOOM_MIN_KEYS
      std::unique_ptr<uint64_t[]> bloom;
      uint64_t bloomMask = 0; // words - 1
#endif
#ifdef VW_JOIN_SEMI
      /// VW_JOIN_SEMI: build key range and the exact key bitmap over it
      std::atomic<int64_t> semiMin{INT64_MAX}, semiMax{INT64_MIN};
      std::unique_ptr<uint32_t[]> semiBits;
      bool semiOn = false;
#endif
      Shared() : found(0), sizeIsSet(false){};
   };

   struct IteratorContinuation
   /// State to continue iteration in next call
   {
      pos_t nextProbe = 0;
      pos_t numProbes = 0;
      runtime::Hashmap::hash_t probeHash;
      runtime::Hashmap::EntryHeader* buildMatch;
      IteratorContinuation()
          : nextProbe(0), numProbes(0), buildMatch(runtime::Hashmap::end()) {}
   } cont;

 private:
   Shared& shared;
   /// Additional state to continue iteration in next call for  joinSelParallel
   struct IterConcurrentContinuation {
      pos_t followup = 0;
      pos_t followupWrite = 0;
      IterConcurrentContinuation() : followup(0), followupWrite(0) {}
   } contCon;
   bool consumed = false;
   std::vector<std::pair<void*, size_t>> allocations;

 public:
   size_t followupBufferSize = 1025;
   pos_t* followupIds;
   runtime::Hashmap::EntryHeader** followupEntries;

 public:
   Hashjoin(Shared& sm);
   pos_t batchSize;
   size_t ht_entry_size;
   Expression buildHash;
   Aggregates buildScatter;
   runtime::Hashmap::EntryHeader* scatterStart;
   Aggregates buildGather;
   runtime::Hashmap::EntryHeader** buildMatches;
   Expression probeHash;
   runtime::Hashmap::hash_t* probeHashes;
   Expression keyEquality;
   pos_t* probeSel = nullptr;
   pos_t* probeMatches;

   /// function which computes join result into buildMatches and probeMatches
   pos_t (Hashjoin::*join)();
   /// computes join result into buildMatches and probeMatches
   pos_t joinAll();
   /// computes join result into buildMatches and probeMatches
   /// Implementation: optimized for long CPU pipelines
   pos_t joinAllParallel();
   // join implementation after Peter's suggestions
   pos_t joinBoncz();
   /// computes join result into buildMatches and probeMatches
   /// Implementation: Using AVX 512 SIMD
#ifndef VW_POS_16
   pos_t joinAllSIMD();
#endif
   /// computes join result into buildMatches and probeMatches, respecting
   /// selection vector probeSel for probe side
   pos_t joinSel();
   /// computes join result into buildMatches and probeMatches, respecting
   /// selection vector probeSel for probe side
   /// Implementation: optimized for long CPU pipelines
   pos_t joinSelParallel();
   /// computes join result into buildMatches and probeMatches, respecting
   /// selection vector probeSel for probe side
   /// Implementation: For SkylakeX using AVX512
#ifndef VW_POS_16
   pos_t joinSelSIMD();
#endif
   /// the plan marked this join a semi join (HashJoinBuilder::semi()): only
   /// existence of the probe key matters. Assumes unique build keys (see
   /// semi()); without VW_JOIN_SEMI it runs as an inner join, equal then.
   bool semiJoin = false;
#ifdef VW_JOIN_SEMI
   /// VW_JOIN_SEMI: a join the plan marked semi (semiJoin) only needs
   /// existence of the probe key. QueryBuilder records whether the keys
   /// qualify (one int32 key per side); after the build, a bitmap over the
   /// build keys' range replaces hash, probe, key equality and gather.
   bool semiOk = true;
   size_t semiBuildKeys = 0;
   size_t semiKeyOffset = 0;
   struct F2_Op* semiProbeDense = nullptr;
   struct F3_Op* semiProbeSel = nullptr;
   bool semiActive = false;
   bool semiCandidate() const {
      return semiJoin && semiOk && semiBuildKeys == 1 &&
             buildGather.ops.empty() &&
             (semiProbeDense || semiProbeSel) && probeHash.ops.size() == 1;
   }
   pos_t semiProbe(size_t n);
   void semiRange();
   void semiSetBits();
#endif
#ifdef VW_JOIN_BLOOM
   /// probes that passed the Bloom filter for the current probe vector:
   /// index into probeHashes and hash, in probe order (bloomOn only)
   std::vector<pos_t> bloomSel;
   std::vector<runtime::Hashmap::hash_t> bloomHashes;
   size_t bloomCount = 0;
   bool bloomOn = false;
   /// vectors to skip the filter for after one where most probes passed
   uint32_t bloomSkip = 0;
   void bloomInsert();
   void bloomFilter(size_t n);
#endif
#ifdef VW_JOIN_FUSED_PROBE
   /// VW_JOIN_FUSED_PROBE: the probe side's single int32 hash op (set by
   /// QueryBuilder::addProbeKey when it is primitives::hash_int32_t_col or
   /// hash_sel_int32_t_col; any second probe key clears fusedOk). The join
   /// then hashes the keys itself instead of evaluating probeHash.
   struct F2_Op* fusedDense = nullptr;
   struct F3_Op* fusedSel = nullptr;
   bool fusedOk = true;
   /// this vector's hashes are computed by joinNewFirstPass (mode B)
   bool fusedCompute = false;
   bool fusedReady() const {
      return fusedOk && (fusedDense || fusedSel) && probeHash.ops.size() == 1;
   }
   void fusedHashFilter(size_t n);
#endif
#ifdef VW_NEW_JOIN
   /// VW_NEW_JOIN (andrew_pseudocode.md): 8 probes at a time, gather the
   /// head entry, split into real / maybe lists, then follow chains
   pos_t joinAllNew();
   pos_t joinSelNew();

 private:
   template <bool Sel> pos_t joinNew();
   template <bool Sel> size_t joinNewFirstPass(pos_t& followupWrite);
   /// the maybe list (probe index, chain successor), appended to the real
   /// list's successors in the followup buffers
   std::vector<pos_t> newMaybeIds;
   std::vector<runtime::Hashmap::EntryHeader*> newMaybeEntries;

 public:
#endif

#ifdef VW_JOIN_DISPATCH
   /// plan fact recorded by QueryBuilder::addBuildKey: the build keys are
   /// read through a selection vector, i.e. the build side is a filtered
   /// subset of its input (the Bloom filter's precondition)
   bool buildKeysSelected = false;

 private:
   /// VW_JOIN_DISPATCH: the probe path, resolved once at the end of the
   /// build from plan facts and the build summary (resolvePaths); next()
   /// then runs probeStep and join per probe vector without re-deciding
   void (Hashjoin::*probeStep)(size_t n) = nullptr;
   void resolvePaths();
   void stepHash(size_t n);
   void stepHashBloom(size_t n);
   void stepFusedBloom(size_t n);
   void stepFusedNew(size_t n);

 public:
#endif
   virtual size_t next() override;
   ~Hashjoin();
};

class HashGroup : public UnaryOperator {
   runtime::Hashmap ht;
   size_t maxFill;
   const size_t initialMapSize = 1024;

 public:
   using hash_t = decltype(ht)::hash_t;
   using deque_t = runtime::PartitionedDeque<1024>;
   using EntryHeader = runtime::Hashmap::EntryHeader;

   size_t nrPartitions;

   size_t vecSize;
   runtime::ResetableAllocator groupStore;

   struct Shared : SharedState {
      std::atomic<size_t> partition;
      runtime::thread_specific<deque_t> spillStorage;
      Shared() : partition(0) {}
   } & shared;

   HashGroup(Shared& shared);
   ~HashGroup();

   std::unique_ptr<runtime::HashmapSmall<pos_t, pos_t>> groupHt;

   /// ------ phase 1: local preaggregation
   /// ------ group lookup

   template <typename T> class GroupLookup {

      /// Find entries in ht for groupHashes.
      /// Found entries are written to htMatches, missing entries
      /// are added to groupsNotFound
      pos_t htLookup(pos_t n, decltype(ht) & ht);
      /// Follows chains in ht for entries in keysNEq
      pos_t htFollow(decltype(ht) & ht);

      HashGroup& parent;

    public:
      GroupLookup(HashGroup& p) : parent(p){};
      std::deque<std::pair<void*, size_t>> allocations;

      size_t ht_entry_size;
      size_t entries_in_ht = 0;
      /// ------- group lookup
      /// Find group entries for all in flight tuples.
      /// Write matches to htMatches
      pos_t findGroups(pos_t n, decltype(ht) & ht);
      /// Buffer which contains hashes of group keys
      hash_t* groupHashes;
      /// Buffer which contains entry pointers after group lookup
      EntryHeader** htMatches;
      /// Tuples that found an aggregate in the hashtable
      pos_t* groupsFound;
      /// Expression to check key equality
      Expression keyEquality;
      /// entries which did not match after key comparison
      SizeBuffer<pos_t>* keysNEq;
      /// Tuples which did not find an aggregate in the hashtable
      SizeBuffer<pos_t>* groupsNotFound;
      /// Pointer to current row format input
      void* rowData;

      /// ------ group creation
      /// Creates missing groups. Returns number of groups created.
      size_t createMissingGroups(decltype(ht) & ht, bool allowResize = false);
      /// Expression to execute partitioning on groupsNotFound
      Expression partitionKeys;
      /// Expression to scatter
      Aggregates buildScatter;
      runtime::Hashmap::EntryHeader* scatterStart;
      /// Partition ends buffer, input to partitionKeys expression
      pos_t* partitionEndsIn;
      /// Partition ends buffer, write target of partitionKeys expression
      pos_t* partitionEndsOut;
      /// Unordered row selector, input to partitionKeys expression
      pos_t* unpartitionedRows;
      /// Reordered row selector, write target of partitionKeys expression
      pos_t* partitionedRows;
      pos_t* groupRepresentatives;
      void getGroupRepresentatives(pos_t nrGroups, pos_t* data,
                                   pos_t* groupEnds, pos_t* out);

      void clearHashtable(decltype(ht) & ht);

    private:
      T* self() { return static_cast<T*>(this); }
   };

   /// Local aggregation phase uses column wise storage as input format for
   /// vectors
   class ColumnGroupLookup : public GroupLookup<ColumnGroupLookup> {
    public:
      hash_t hashForTuple(size_t i) { return groupHashes[i]; }
      ColumnGroupLookup(HashGroup& p) : GroupLookup<ColumnGroupLookup>(p){};
   };

   // Global aggregation reads from PartitionedDeque, which uses row wise
   // storage format
   class RowGroupLookup : public GroupLookup<RowGroupLookup> {
    public:
      size_t rowSize;
      hash_t hashForTuple(size_t i) {
         return *addBytes(reinterpret_cast<hash_t*>(rowData), rowSize * i);
      }
      RowGroupLookup(HashGroup& p) : GroupLookup<RowGroupLookup>(p){};
   };

   ColumnGroupLookup preAggregation;
   /// update aggregates
   Aggregates updateGroups;
#ifdef VW_AGGR_FUSED
   /// VW_AGGR_FUSED: updateGroups' ops when all are int64 SUM / COUNT(*)
   /// (recorded by QueryBuilder::HashGroupBuilder::addValue). Their current
   /// arguments are read at run time; updateGroupsFused then updates every
   /// aggregate of a row's entry in one pass instead of one pass each.
   struct FusedAggr {
      enum Kind : uint8_t { Col, SelCol, Count } kind;
      Op* op;
      size_t offset;
   };
   std::vector<FusedAggr> fusedAggrs;
   bool fusable = true;
   std::vector<pos_t> identitySel;
   std::vector<int64_t> ones;
   void updateGroupsFused(pos_t n);
#endif

   /// ------ phase 2: global aggregation
   RowGroupLookup globalAggregation;
   pos_t findGroupsFromPartition(void* data, size_t n);
#ifdef VW_GROUP_GLOBAL_DIRECT
   /// VW_GROUP_GLOBAL_DIRECT: set by QueryBuilder when every key compares
   /// correctly with memcmp (fixed-width scalar types; Char<N> padding bytes
   /// are undefined); then the global phase uses globalFindOrCreate
   bool globalDirectOk = true;
   void globalFindOrCreate(void* data, size_t n);
#endif
   Aggregates updateGroupsFromPartition;

   /// ------ produce result
   /// Gather primitives to produce vectors after
   /// global grouping is done
   Aggregates gatherGroups;
#ifdef VW_GROUP_HAVING
   /// VW_GROUP_HAVING (HashGroupBuilder::having): an output block first
   /// gathers the condition's input column (havingInput), evaluates the
   /// condition into havingSel, and gathers the other output columns only
   /// for the passing groups (gatherGroups, re-pointed at havingSel); the
   /// input column is then compacted in place. Blocks with no passing group
   /// are skipped.
   std::unique_ptr<Op> havingInput;
   std::unique_ptr<Expression> havingCondition;
   pos_t* havingSel = nullptr;
   void* havingInputTarget = nullptr;
   size_t havingInputSize = 0;
   /// output element size of each gatherGroups op, in order
   std::vector<size_t> gatherSizes;
#endif

   struct Continuation
   /// state for control flow
   {
      decltype(globalAggregation.allocations.begin()) iter;
      decltype(nrPartitions) partition = 0;
      /// Is the input to this operator already consumed?
      bool consumed = false;
      /// Does the current partition need aggregation or can the result already
      /// be passed on to the next iterator?
      bool partitionNeedsAggregation = true;
   } cont;

   virtual size_t next() override;

   struct KeyColumn {
      uint32_t size = 0;
      uint32_t offset = 0;
      void* data = nullptr;
      /// Selection vector for this key column: row i reads data[sel[i]].
      /// nullptr for keys that are already dense (e.g. join output buffers),
      /// which are read as data[i]. Keys of one group-by can mix both kinds
      /// (TPC-H Q3: l_orderkey via sel, o_orderdate/o_shippriority dense).
      pos_t* sel = nullptr;
   };

   uint32_t totalKeySize = 0;
   std::deque<KeyColumn> keyColumns;
   std::vector<char> packedKeys;
#ifdef VW_GROUP_NO_CONCAT
   /// packed keys of the current vector: packedKeys, or the key column
   /// itself for a single dense key
   char* keyData = nullptr;
#endif
   pos_t* selVec = nullptr;

#ifdef VW_USE_CRC32
   runtime::CRC32Hash hashFn;
#else
   runtime::MurMurHash hashFn;
#endif

   void Concat(pos_t n);
   template <typename T> void Concat_T(pos_t n, const KeyColumn& col);

   void Hash(pos_t n);
   template <typename T> void Hash_T(pos_t n);

   void Lookup(pos_t n);
   template <typename T> void Lookup_T(pos_t n);
#ifdef VW_GROUP_DISPATCH
   /// VW_GROUP_DISPATCH: the pre-aggregation path is chosen once per operator
   /// from what the plan fixes for its lifetime (number of key columns, key
   /// width, whether the key comes through a selection vector, the
   /// aggregates), instead of per-vector switches on totalKeySize
   enum KeyMode { kSingle, kSingleSel, kPacked };
   using VecStep = void (HashGroup::*)(pos_t n);
   VecStep keyStep = nullptr;  ///< keys -> preAggregation.htMatches
   VecStep aggStep = nullptr;  ///< aggregate update
   std::vector<void (HashGroup::*)(pos_t, const KeyColumn&)> concatSteps;
   bool pathsResolved = false;
   void resolvePaths();
   template <typename T, int Mode, bool Runs> void keyStepT(pos_t n);
#ifdef VW_GROUP_RUN_HEADS
   /// VW_GROUP_RUN_HEADS: per vector, the positions where the key differs
   /// from the previous row's (run heads), a 0/1 head flag per row, and each
   /// run's entry; the lookup runs only for heads
   std::vector<pos_t> runHeads;
   std::vector<uint8_t> runFlags;
   std::vector<EntryHeader*> runEntries;
#endif
   void concatResolved(pos_t n);
   void aggEvaluate(pos_t n) { updateGroups.evaluate(n); }
#endif
#ifdef VW_GROUP_BATCH_CREATE
   /// VW_GROUP_BATCH_CREATE: pre-aggregation entries are carved from one
   /// block per flush period (maxFill + vecSize entries, so a vector never
   /// runs out), consecutively, so one vector's new groups can be
   /// initialized by a single buildScatter call
   EntryHeader* newBlock = nullptr;
   size_t newUsed = 0;
   size_t newCap = 0;
#endif

 private:
   void clearHashtable();
};

template <typename T>
pos_t INTERPRET_SEPARATE
HashGroup::GroupLookup<T>::htLookup(pos_t n, runtime::Hashmap& ht) {
   pos_t found = 0;
   for (size_t i = 0; i < n;) {
      auto hash = self()->hashForTuple(i);
      auto el = ht.find_chain(hash);
      if (el != ht.end()) {
         if (el->hash == hash) {
            htMatches[i] = el;
            groupsFound[found++] = i;
            goto nextChain;
         }
         for (el = el->next; el != ht.end(); el = el->next)
            if (el->hash == hash) {
               htMatches[i] = el;
               groupsFound[found++] = i;
               goto nextChain;
            }
      }
      groupsNotFound->push_back(i);
   nextChain:
      ++i;
   }
   return found;
}

template <typename T>
pos_t HashGroup::GroupLookup<T>::htFollow(runtime::Hashmap& ht)
/// follows chains in keysNEq
{
   pos_t found = 0;
   for (size_t i = 0, end = keysNEq->size(); i < end;) {
      auto idx = keysNEq->operator[](i);
      for (auto e = htMatches[idx]->next; e != ht.end(); e = e->next) {
         auto hash = self()->hashForTuple(idx);
         if (e->hash == hash) {
            htMatches[idx] = e;
            groupsFound[found++] = idx;
            goto nextChain;
         }
         groupsNotFound->push_back(i);
      }
   nextChain:
      ++i;
   }
   return found;
}

template <typename T>
pos_t INTERPRET_SEPARATE
HashGroup::GroupLookup<T>::findGroups(pos_t n, runtime::Hashmap& ht) {
   keysNEq->clear();
   groupsNotFound->clear();
   auto found = htLookup(n, ht);
   auto keysEqual = keyEquality.evaluate(found);
   while (keysNEq->size()) {
      found = htFollow(ht);
      if (!found) break;
      keysNEq->clear();
      keysEqual += keyEquality.evaluate(found);
   }
   return keysEqual;
}

template <typename T>
void HashGroup::GroupLookup<T>::getGroupRepresentatives(pos_t nrGroups,
                                                        pos_t* sel,
                                                        pos_t* groupEnds,
                                                        pos_t* out) {
   for (pos_t i = 0, groupId = 0; i < nrGroups; groupId = groupEnds[i++])
      out[i] = sel[groupId];
}

template <typename T>
size_t INTERPRET_SEPARATE HashGroup::GroupLookup<T>::createMissingGroups(
    runtime::Hashmap& ht, bool allowResize) {
   auto n = groupsNotFound->size();
   if (!n) return 0;
   // find groups
   partitionEndsIn[0] = n;
   auto groups = partitionKeys.evaluate(1);
   entries_in_ht += groups;
   // reserve memory for entries
   // auto alloc = compat::aligned_alloc(16, groups * ht_entry_size);
   auto alloc = parent.groupStore.allocate(groups * ht_entry_size);
   if (!alloc) throw std::runtime_error("malloc failed");
   allocations.push_back(std::make_pair(alloc, groups));
   // write representative row for each group into groupRepresentatives
   getGroupRepresentatives(groups, partitionedRows, partitionEndsOut,
                           groupRepresentatives);
   // scatter hash, keys and aggregate initializers into ht entries
   scatterStart = reinterpret_cast<decltype(scatterStart)>(alloc);
   buildScatter.evaluate(groups);
   using header_t = runtime::Hashmap::EntryHeader;
   // insert groups into ht
   if (allowResize && entries_in_ht > parent.maxFill) {
      // clear hashtable and prepare it for new size
      parent.maxFill = ht.setSize(entries_in_ht * 2);
      // reinsert all entries
      for (auto& block : allocations)
         ht.insertAll<false>(reinterpret_cast<header_t*>(block.first),
                             block.second, ht_entry_size);
   } else {
      auto& block = allocations.back();
      ht.insertAll<false>(reinterpret_cast<header_t*>(block.first),
                          block.second, ht_entry_size);
   }
   // write pointers back to htMatches
   pos_t pStart = 0;
   runtime::Hashmap::EntryHeader* entry =
       reinterpret_cast<runtime::Hashmap::EntryHeader*>(alloc);
   for (pos_t p = 0; p < groups; ++p) { // each partition
      for (pos_t i = pStart, end = partitionEndsOut[p]; i < end; ++i)
         htMatches[partitionedRows[i]] = entry;
      entry = addBytes(entry, ht_entry_size);
      pStart = partitionEndsOut[p];
   }
   return groups;
}

template <typename T>
void INTERPRET_SEPARATE
HashGroup::GroupLookup<T>::clearHashtable(runtime::Hashmap& ht) {
   ht.clear();
   entries_in_ht = 0;
   parent.groupStore.reset();
   // for (auto& alloc : allocations) free(alloc.first);
   allocations.clear();
}
} // namespace vectorwise
