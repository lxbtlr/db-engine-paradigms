#pragma once
#include "Operators.hpp"
#include "common/runtime/Database.hpp"
#include "vectorwise/VectorAllocator.hpp"
#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

namespace vectorwise {

class QueryBuilder {
 public:
   runtime::GlobalPool* previous;

 protected:
   std::stack<std::unique_ptr<Operator>> operatorStack;
   runtime::Database& db;
   SharedStateManager& operatorState;
   VectorAllocator vecs;
   std::unordered_map<size_t, std::pair<size_t, void*>> buffers;
   /// uniqueness facts the plan establishes, for deriving semi joins:
   /// buffers holding a unique column (a group-by's single key output)
   std::unordered_set<void*> uniqueBuffers;
   /// selection buffers that may list a row more than once (outputs of joins
   /// whose build side is not unique); other selections list rows once
   std::unordered_set<void*> repeatingSelections;

   struct DataStorage
   /// handle for data sources, e.g. base table columns or cache buffers
   {
      enum BufferSpec { Buffer, Column, Value, None } buf = None;
      size_t dataSize;
      void* data = nullptr;
      class Scan* scan = nullptr;
      std::string attribute;
      /// Column: the relation it belongs to (declared keys, uniqueBuild)
      const runtime::Relation* rel = nullptr;
      void registerDS(void** location);
      void registerDS(pos_t** location);
      operator void*() const;
      operator pos_t*() const;
   };
   using DS = DataStorage;

   struct ResultBuilder {
      using RB = ResultBuilder;
      QueryBuilder& base;
      ResultWriter& resultWriter;
      std::unique_ptr<ResultWriter> resultWriterOwning;
      RB& addValue(std::string name, DS buffer);
      void finalize();
   };

   struct ScanBuilder {
      class Scan& scan;
      runtime::Relation& rel;
   };

   struct ProjectionBuilder {
      QueryBuilder& base;
      class Project& project;
      ProjectionBuilder& addExpression(std::unique_ptr<Expression>&& exp);
   };

   struct HashJoinBuilder {
      QueryBuilder& base;
      bool probeHasSelection = false;
      std::deque<size_t> keyOffsets;
      void* buildHashBuffer = nullptr;
      void* probeHashBuffer = nullptr;
      Hashjoin* join;
      /// build key columns and the selection each is read through (nullptr:
      /// dense), for uniqueBuild
      struct BuildKey {
         DS col;
         void* sel;
      };
      std::vector<BuildKey> buildKeys;
      /// selections pushProbeSelVector derives from this join's output
      std::vector<void*> pushedTargets;
      bool semiAsserted = false;
      HashJoinBuilder(QueryBuilder& b);
      /// derives the semi join (see semi()); throws if semi() was asserted
      /// and the derivation fails
      ~HashJoinBuilder() noexcept(false);
      using B = HashJoinBuilder;

      B& addBuildKey(DS col, primitives::F2 hash, primitives::FScatter scatter);
      B& addBuildKey(DS col, DS sel, primitives::F3 hash,
                     primitives::FScatterSel scatter);
      B& addProbeKey(DS col, primitives::F2 hash, primitives::EQCheck eq);
      B& addProbeKey(DS col, DS sel, primitives::F3 hash,
                     primitives::EQCheck eq);
      B& addProbeKey(DS col, DS sel, primitives::F3 hash, DS selEq,
                     primitives::EQCheck eq);
      B& addBuildValue(DS source, primitives::FScatter scatter, DS target,
                       primitives::FGather gather);
      B& addBuildValue(DS source, DS sel, primitives::FScatterSel scatter,
                       DS target, primitives::FGather gather);
      B&
      setProbeSelVector(DS vec,
                        pos_t (Hashjoin::*join)() = &Hashjoin::joinSelParallel);
      B& pushProbeSelVector(DS sel, DS target);
      /// Assert that this join is a semi join. The builder derives it for
      /// every join (~HashJoinBuilder): when the build keys are unique
      /// (QueryBuilder::uniqueBuild: declared primary keys, a group-by's key,
      /// through selections that never repeat a row) and no build column is
      /// used, each probe row matches at most one build row, so the inner
      /// join and the semi join return the same rows and Hashjoin::semiJoin
      /// is set. semi() makes plan building fail (std::runtime_error) when
      /// that derivation does not hold, so a plan cannot claim a semi join
      /// the schema and plan do not prove. Q18's IN subquery and the
      /// primary-key joins of Q3, Q5, Q9 (Hyper's plans use Hashset +
      /// contains() for the same joins).
      B& semi();
   };

   struct HashGroupBuilder {
      QueryBuilder& base;
      vectorwise::HashGroup* group;

      struct Lookup {
         pos_t* partitionEndsIn;
         pos_t* partitionEndsOut;
         pos_t* unpartitionedRows;
         pos_t* partitionedRows;
      } localLookup, globalLookup;

      HashGroupBuilder(QueryBuilder& base);
      /// output buffers of the key columns; a single one is unique
      std::vector<void*> keyOutputs;

      using B = HashGroupBuilder;
      B& addKey(DS col, primitives::F2 hash,
                /**** global aggr *****/
                primitives::NEQCheck eq,
                primitives::FPartitionByKey partitionByKey,
                primitives::FScatterSel scatter,
                /**** global aggr *****/
                primitives::NEQCheckRow eqG,
                primitives::FPartitionByKeyRow partitionByKeyG,
                primitives::FScatterSelRow scatterG,
                /**** output *****/
                primitives::FGatherVal gather, DS out);
      B& addKey(
          /**** input *****/
          DS col, DS sel, primitives::F3 hash,
          /**** local aggregation *****/
          primitives::NEQCheckSel eq,
          primitives::FPartitionByKeySel partitionByKey, DS selScat,
          primitives::FScatterSel scatter,
          /**** global aggregation *****/
          primitives::NEQCheckRow eqG,
          primitives::FPartitionByKeyRow partitionByKeyG,
          primitives::FScatterSelRow scatterG,
          /**** output *****/
          primitives::FGatherVal gather, DS out);
      B& pushKeySelVec(DS sel, DS outBuf);
      B& addValue(DS col, primitives::FAggrInit aggrInit,
                  primitives::FAggr aggr, primitives::FAggrRow aggrGlobal,
                  primitives::FGatherVal gather, DS out);
      B& addValue(DS col, DS sel, primitives::FAggrInit aggrInit,
                  primitives::FAggrSel aggr, primitives::FAggrRow aggrGlobal,
                  primitives::FGatherVal gather, DS out);
      B& padToAlign(size_t align);
#ifdef VW_GROUP_HAVING
      /// SQL HAVING: keep only the groups whose output passes condition, a
      /// selection Expression over input (one of this group-by's outputs)
      /// that writes the passing positions to selection. Call after every
      /// addKey/addValue. Consumers see only the passing groups, densely.
      B& having(DS input, std::unique_ptr<vectorwise::Expression>&& condition,
                DS selection);
#endif
      ~HashGroupBuilder();
   };

   struct ExpressionBuilder {
      std::unique_ptr<vectorwise::Expression> expression;
      using DS = DataStorage;
      ExpressionBuilder& addOp(primitives::F1 op, DS a);
      ExpressionBuilder& addOp(primitives::F2 op, DS a, DS b);
      ExpressionBuilder& addOp(primitives::F3 op, DS a, DS b, DS c);
      ExpressionBuilder& addOp(primitives::F4 op, DS a, DS b, DS c, DS d);
      operator std::unique_ptr<vectorwise::Expression>();
      operator std::unique_ptr<vectorwise::Aggregates>();
   };

   QueryBuilder(runtime::Database& db_, SharedStateManager& s,
                size_t vSize = 1024)
       : db(db_), operatorState(s), vecs(vSize) {}

   size_t opNr = 0;
   size_t onceNr = 0;
   size_t nextOpNr();
   size_t nextOnceNr();

   ResultBuilder Result();
   ScanBuilder Scan(std::string relation);
   template <typename PAYLOAD>
   void Debug(std::function<void(size_t, PAYLOAD&)> step,
              std::function<void(PAYLOAD&)> finish);
   void DebugCounter(std::string message);
   void Select(std::unique_ptr<Expression>&& exp);
   ProjectionBuilder Project();
   void FixedAggregation(std::unique_ptr<Aggregates>&& aggrs);
   HashJoinBuilder
   HashJoin(DS probeMatches,
            pos_t (Hashjoin::*join)() = &Hashjoin::joinAllParallel);
   HashGroupBuilder HashGroup();
   /// whether a join's build keys are unique, from declared keys and the
   /// plan (no data): all keys read through the same selection, which never
   /// lists a row twice, and either a group-by's single key output, or
   /// columns of one relation that cover its declared primary key
   bool uniqueBuild(const std::vector<HashJoinBuilder::BuildKey>& keys) const;

   ~QueryBuilder();

   ExpressionBuilder Expression();

   DS Buffer(size_t nr, size_t entrySize);
   DS Buffer(size_t nr);
   DS Column(ScanBuilder& scan, std::string attribute);
   DS Value(void*);

   void pushOperator(std::unique_ptr<Operator>&& op);
   std::unique_ptr<Operator> popOperator();
};

template <typename PAYLOAD>
void QueryBuilder::Debug(std::function<void(size_t, PAYLOAD&)> step,
                         std::function<void(PAYLOAD&)> finish) {

   auto& shared = operatorState.get<typename DebugOperator<PAYLOAD>::Shared>(nextOpNr());
   auto debug = std::make_unique<DebugOperator<PAYLOAD>>(
       shared, std::move(step), std::move(finish));
   debug->child = popOperator();
   pushOperator(std::move(debug));
}
} // namespace vectorwise
