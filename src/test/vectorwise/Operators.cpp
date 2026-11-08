#include "../TPCH.hpp"
#include "benchmarks/tpch/Queries.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Import.hpp"
#include "common/runtime/MemoryPool.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/Query.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace vectorwise;
using namespace std;

namespace operatortest {

struct SelectTest : public TPCH, public Query, public QueryBuilder {
   runtime::GlobalPool pool;
   SelectTest() : Query(), QueryBuilder(TPCH::getDB(), shared) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   }
};

TEST_F(SelectTest, buildingEQ) {
   enum { sel_cust };

   // constants
   std::string building = "BUILDING";
   types::Char<10> c1 =
       types::Char<10>::castString(building.data(), building.size());

   int64_t count = 0;

   auto customer = Scan("customer");
   Select(Expression().addOp(
       primitives::sel_equal_to_Char_10_col_Char_10_val_bf,        //
       Buffer(sel_cust, sizeof(pos_t)),                            //
       Column(customer, "c_mktsegment"),                           //
       Value(&c1)));                                               //
   FixedAggregation(Expression()                                   //
                        .addOp(primitives::aggr_static_count_star, //
                               Value(&count)));

   auto root = popOperator();
   auto n = root->next();
   ASSERT_EQ(pos_t(1), n);
   EXPECT_EQ(30142, count);
}

struct SimpleJoinBuilder : public Query, public vectorwise::QueryBuilder {
   enum { buildValue, probe_matches };
   struct Result {
      int32_t* r;
      std::unique_ptr<vectorwise::Operator> rootOp;
   };
   runtime::GlobalPool pool;
   SimpleJoinBuilder(runtime::Database& db, size_t v = 1024)
       : Query(), QueryBuilder(db, shared, v) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   }
   unique_ptr<Result> getQuery() {
      auto r = make_unique<Result>();
      auto build = Scan("build");
      auto probe = Scan("probe");
      HashJoin(Buffer(probe_matches, sizeof(pos_t)))
          .addBuildKey(Column(build, "k"), conf.hash_int32_t_col(),
                       primitives::scatter_int32_t_col)
          .addProbeKey(Column(probe, "b"), conf.hash_int32_t_col(),
                       primitives::keys_equal_int32_t_col)
          .addBuildValue(Column(build, "v"), primitives::scatter_int32_t_col,
                         Buffer(buildValue, sizeof(int32_t)),
                         primitives::gather_col_int32_t_col);
      r->r = reinterpret_cast<int32_t*>(Buffer(buildValue).data);
      r->rootOp = popOperator();
      return r;
   }
};

template <typename T>
void assertAllContained(T* d, size_t n, unordered_multiset<T> expected) {
   for (size_t i = 0; i < n; i++) {
      auto e = expected.find(d[i]);
      // TODO: message
      if (e == expected.end()) { ASSERT_TRUE(false); }
      expected.erase(e);
   }
   ASSERT_EQ(expected.size(), 0ull);
}

TEST(Join, simpleJoin) {

   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 3, 4, 8};
   db["build"].insert("v", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{101, 103, 104, 108};
   db["probe"].insert("b", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{88, 1, 1, 17, 4, // 1 2 4
                            1,  2, 3, 4,     // 5 7 8
                            1,  2, 3, 4,     // 9 11 12
                            1,  2, 3, 4,     // 13 14 16
                            1,  2, 3, 4,     // 17 19 20
                            1,  2, 3, 4};    // 21 23 24
   db["build"].nrTuples = 4;
   db["probe"].nrTuples = 25;

   SimpleJoinBuilder b(db);
   auto query = b.getQuery();
   auto n = query->rootOp->next();
   ASSERT_EQ(pos_t(18), n);
   auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
   ASSERT_NE(nullptr, join);
   // check if correct elements of probe side were determined
   assertAllContained(join->probeMatches, n,
                      {1, 2, 4,    //
                       5, 7, 8,    //
                       9, 11, 12,  //
                       13, 15, 16, //
                       17, 19, 20, //
                       21, 23, 24});

   unordered_multiset<pos_t> vMatches{1, 1, 4, //
                                      1, 3, 4, //
                                      1, 3, 4, //
                                      1, 3, 4, //
                                      1, 3, 4, //
                                      1, 3, 4};
   for (size_t i = 0; i < n; i++) {
      auto& el = *addBytes(reinterpret_cast<int32_t*>(join->buildMatches[i++]),
                           sizeof(runtime::Hashmap::EntryHeader));
      auto e = vMatches.find(el);
      ASSERT_NE(e, vMatches.end());
   }
   // check if correct build values were gathered
   assertAllContained(query->r, n,
                      {
                          101,
                          101,
                          104, //
                          101,
                          103,
                          104, //
                          101,
                          103,
                          104, //
                          101,
                          103,
                          104, //
                          101,
                          103,
                          104, //
                          101,
                          103,
                          104,
                      });
}

TEST(Join, simpleJoinWithSmallBuffers) {
   /// Tests if join works when not all data fits into the buffers
   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 3, 4, 8};
   db["build"].insert("v", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{101, 103, 104, 108};
   db["probe"].insert("b", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{88, 1, 1, 17, 4};
   db["build"].nrTuples = 4;
   db["probe"].nrTuples = 5;

   SimpleJoinBuilder b(db, 2);
   auto query = b.getQuery();
   std::vector<int32_t> expectedKeys = {1, 1, 4};
   std::vector<int32_t> expectedBuildValues = {101, 101, 104};
   size_t offset = 0;
   size_t found = 0;
   while (auto n = query->rootOp->next()) {
      found += n;
      auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
      ASSERT_NE(nullptr, join);
      int i = 0;
      // check if correct keys of build side were found
      for (unsigned j = 0; j < n; ++j)
         ASSERT_EQ(
             expectedKeys[offset + j],
             *addBytes(reinterpret_cast<int32_t*>(join->buildMatches[i++]),
                       sizeof(runtime::Hashmap::EntryHeader)));
      // check if correct build values were gathered
      i = 0;
      for (unsigned j = 0; j < n; ++j)
         ASSERT_EQ(expectedBuildValues[offset + j], query->r[i++]);
      offset += n;
   }

   ASSERT_EQ(expectedKeys.size(), found);
}
TEST(Join, simpleJoinWithResultOverflow) {
   /// Tests if join works when not all data fits into the buffers
   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 1, 1, 1, 1, 3, 4, 8};
   db["build"].insert("v", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{101, 101, 101, 101, 101, 103, 104, 108};
   db["probe"].insert("b", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{88, 1, 26, 4, 9};
   db["build"].nrTuples = 8;
   db["probe"].nrTuples = 5;

   SimpleJoinBuilder b(db, 2);
   auto query = b.getQuery();
   std::unordered_multiset<int32_t> expectedKeys = {1, 1, 1, 1, 1, 4};
   std::unordered_multiset<int32_t> expectedBuildValues = {101, 101, 101,
                                                           101, 101, 104};
   size_t found = 0;

   vector<int32_t> keys;
   vector<int32_t> vals;
   while (auto n = query->rootOp->next()) {
      found += n;
      ASSERT_LE(n, pos_t(2));
      auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
      ASSERT_NE(nullptr, join);

      for (unsigned i = 0; i < n; ++i)
         keys.push_back(
             *addBytes(reinterpret_cast<int32_t*>(join->buildMatches[i]),
                       sizeof(runtime::Hashmap::EntryHeader)));
      for (unsigned i = 0; i < n; ++i) vals.push_back(query->r[i]);
   }

   // check if correct keys of build side were found
   assertAllContained(keys.data(), keys.size(), expectedKeys);
   // check if correct build values were gathered
   assertAllContained(vals.data(), vals.size(), expectedBuildValues);
   ASSERT_EQ(expectedKeys.size(), found);
}

struct JoinBuildSelectBuilder : public Query, private vectorwise::QueryBuilder {
   enum { buildValue, sel_key, probe_matches };
   struct Result {
      int64_t bound = 103;
      int64_t* r;
      std::unique_ptr<vectorwise::Operator> rootOp;
   };
   runtime::GlobalPool pool;
   JoinBuildSelectBuilder(runtime::Database& db)
       : Query(), QueryBuilder(db, shared) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   }
   unique_ptr<Result> getQuery() {
      auto r = make_unique<Result>();
      auto build = Scan("build");
      Select(Expression().addOp(primitives::sel_greater_int64_t_col_int64_t_val,
                                Buffer(sel_key, sizeof(int64_t)), //
                                Column(build, "v"),               //
                                Value(&r->bound)));
      auto probe = Scan("probe");
      HashJoin(Buffer(probe_matches, sizeof(pos_t)))
          .addBuildKey(Column(build, "k"), Buffer(sel_key),
                       primitives::hash_sel_int64_t_col,
                       primitives::scatter_sel_int64_t_col)
          .addProbeKey(Column(probe, "b"), primitives::hash_int64_t_col,
                       primitives::keys_equal_int64_t_col)
          .addBuildValue(Column(build, "v"), Buffer(sel_key),
                         primitives::scatter_sel_int64_t_col,
                         Buffer(buildValue, sizeof(int64_t)),
                         primitives::gather_col_int64_t_col);
      r->r = reinterpret_cast<int64_t*>(Buffer(buildValue).data);
      r->rootOp = popOperator();
      return r;
   }
};

TEST(Join, joinBuildSelection) {
   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 3, 4, 8, 22, 33, 44, 55, 34};
   db["build"].insert("v", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{101, 103, 104, 108, 122, 133, 144, 155, 134};
   db["probe"].insert("b", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{88, 1, 1, 17, 4, 3, 44, 44};
   db["build"].nrTuples = 9;
   db["probe"].nrTuples = 8;

   JoinBuildSelectBuilder b(db);
   auto query = b.getQuery();
   auto n = query->rootOp->next();
   ASSERT_EQ(pos_t(3), n);
   auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
   ASSERT_NE(nullptr, join);
   // check if correct elements of probe side were determined
   assertAllContained(join->probeMatches, n, {4, 6, 7});
   // check if correct keys of build side were found

   unordered_multiset<pos_t> vMatches{4, 44, 44};
   for (size_t i = 0; i < n; i++) {
      auto& el = *addBytes(reinterpret_cast<int32_t*>(join->buildMatches[i++]),
                           sizeof(runtime::Hashmap::EntryHeader));
      auto e = vMatches.find(el);
      ASSERT_NE(e, vMatches.end());
   }
   // check if correct build values were gathered
   assertAllContained(query->r, n, {104, 144, 144});
}

struct ProbeSelectBuilder : public Query, private vectorwise::QueryBuilder {
   enum { buildValue, sel_probe, probe_matches };
   struct Result {
      int64_t bound = 4;
      std::unique_ptr<vectorwise::Operator> rootOp;
   };
   runtime::GlobalPool pool;
   ProbeSelectBuilder(runtime::Database& db, size_t v = 1024)
       : Query(), QueryBuilder(db, shared, v) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   }
   unique_ptr<Result> getQuery() {
      auto r = make_unique<Result>();
      auto build = Scan("build");
      auto probe = Scan("probe");
      Select(Expression().addOp(primitives::sel_less_int64_t_col_int64_t_val,
                                Buffer(sel_probe, sizeof(pos_t)),
                                Column(probe, "b"), Value(&r->bound)));
      HashJoin(Buffer(probe_matches, sizeof(pos_t)))
          .setProbeSelVector(Buffer(sel_probe))
          .addBuildKey(Column(build, "k"), primitives::hash_int64_t_col,
                       primitives::scatter_int64_t_col)
          .addProbeKey(Column(probe, "b"), Buffer(sel_probe),
                       primitives::hash_sel_int64_t_col,
                       primitives::keys_equal_int64_t_col);
      r->rootOp = popOperator();
      return r;
   }
};

TEST(Join, joinProbeSelection) {

   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 3, 4, 8};
   db["probe"].insert("b", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{88, 8, 1, 1, 17, 4, 3, 3, 3,
                            3,  3, 3, 3, 3,  3, 3, 3, 3};
   db["build"].nrTuples = 4;
   db["probe"].nrTuples = 18;

   ProbeSelectBuilder b(db);
   auto query = b.getQuery();
   auto n = query->rootOp->next();
   ASSERT_EQ(pos_t(14), n);
   auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
   ASSERT_NE(nullptr, join);
   // check if correct elements of probe side were determined
   assertAllContained(join->probeMatches, n,
                      {2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
   // check if correct keys of build side were found
   unordered_multiset<pos_t> vMatches{1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
   for (size_t i = 0; i < n; i++) {
      auto& el = *addBytes(reinterpret_cast<int32_t*>(join->buildMatches[i++]),
                           sizeof(runtime::Hashmap::EntryHeader));
      auto e = vMatches.find(el);
      ASSERT_NE(e, vMatches.end());
   }
}
TEST(Join, joinProbeSelectionAndResultOverflow) {

   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 1, 1, 1, 1, 3, 4, 8};
   db["probe"].insert("b", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{88, 8, 16, 1, 17, 4};
   db["build"].nrTuples = 8;
   db["probe"].nrTuples = 6;

   ProbeSelectBuilder b(db, 2);
   auto query = b.getQuery();
   std::vector<int64_t> expectedKeys = {1, 1, 1, 1, 1};
   size_t offset = 0;
   size_t found = 0;
   while (auto n = query->rootOp->next()) {
      found += n;
      ASSERT_LE(n, pos_t(2));
      auto join = dynamic_cast<Hashjoin*>(query->rootOp.get());
      ASSERT_NE(nullptr, join);
      int i = 0;
      i = 0;
      // check if correct keys of build side were found
      for (unsigned j = 0; j < n; ++j)
         ASSERT_EQ(
             expectedKeys[offset + j],
             *addBytes(reinterpret_cast<int64_t*>(join->buildMatches[i++]),
                       sizeof(runtime::Hashmap::EntryHeader)));
      offset += n;
   }

   ASSERT_EQ(expectedKeys.size(), found);
}

class HashGroupT : public ::testing::Test, public Query, public QueryBuilder {

 protected:
   runtime::Database db;
   runtime::GlobalPool pool;
   HashGroupT() : Query(), QueryBuilder(db, shared) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   };
};

TEST_F(HashGroupT, simpleGroup) {

   enum { grouped_k1, grouped_k2, aggregated_v };
   auto& rel = db["t"];
   rel.insert("k1", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 1, 1, 1, 1, 3, 4, 8};
   rel.insert("k2", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 2, 1, 1, 2, 9, 17, 4};
   rel.insert("v", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{4, 8, 16, 1, 2, 3, 17, 4};
   rel.nrTuples = 8;

   auto t = Scan("t");
   HashGroup()
       .addKey(Column(t, "k1"), primitives::hash_int64_t_col,
               primitives::keys_not_equal_int64_t_col,
               primitives::partition_by_key_int64_t_col,
               primitives::scatter_sel_int64_t_col,
               primitives::keys_not_equal_row_int64_t_col,
               primitives::partition_by_key_row_int64_t_col,
               primitives::scatter_sel_row_int64_t_col,
               primitives::gather_val_int64_t_col,
               Buffer(grouped_k1, sizeof(int64_t)))
       .addKey(Column(t, "k2"), primitives::hash_int32_t_col,
               primitives::keys_not_equal_int32_t_col,
               primitives::partition_by_key_int32_t_col,
               primitives::scatter_sel_int32_t_col,
               primitives::keys_not_equal_row_int32_t_col,
               primitives::partition_by_key_row_int32_t_col,
               primitives::scatter_sel_row_int32_t_col,
               primitives::gather_val_int32_t_col,
               Buffer(grouped_k2, sizeof(int32_t)))
       .addValue(Column(t, "v"), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(aggregated_v, sizeof(int64_t)));
   std::map<std::tuple<int64_t, int32_t>, int64_t> expectedGroups = {
       {make_tuple(1, 1), 21},
       {make_tuple(1, 2), 10},
       {make_tuple(3, 9), 3},
       {make_tuple(4, 17), 17},
       {make_tuple(8, 4), 4}};

   auto root = popOperator();
   size_t found = 0;
   while (auto n = root->next()) {
      found += n;
      auto keys1 = (int64_t*)Buffer(grouped_k1).data;
      auto keys2 = (int32_t*)Buffer(grouped_k2).data;
      auto aggrs = (int64_t*)Buffer(aggregated_v).data;
      for (size_t i = 0; i < n; ++i) {
         int64_t ex = expectedGroups[std::make_tuple(keys1[i], keys2[i])];
         ASSERT_EQ(aggrs[i], ex);
      }
   }
   ASSERT_EQ(found, size_t(5));
}

// 900 groups of a 12-byte key (int64, int32) over 20000 rows: entries are
// created, found again and merged by the global phase; with
// VW_GROUP_KEY_PAD the key is padded to 16 bytes, whose padding must stay
// zero in every entry for whole-key comparison to find the groups. Kept
// under the pre-aggregation fill limit (1433): above it this harness loses
// aggregates also without padding (2026-11-07, not investigated)
TEST_F(HashGroupT, compoundKeyManyGroups) {
   enum { grouped_k1, grouped_k2, aggregated_v };
   constexpr size_t rows = 20000;
   std::vector<int64_t> k1(rows), v(rows);
   std::vector<int32_t> k2(rows);
   std::map<std::tuple<int64_t, int32_t>, int64_t> expected;
   std::mt19937_64 rng(5);
   for (size_t i = 0; i < rows; ++i) {
      k1[i] = int64_t(rng() % 30) - 15;
      k2[i] = int32_t(rng() % 30);
      v[i] = int64_t(rng() % 1000);
      expected[std::make_tuple(k1[i], k2[i])] += v[i];
   }
   auto& rel = db["t"];
   rel.insert("k1", make_unique<algebra::BigInt>()) = std::move(k1);
   rel.insert("k2", make_unique<algebra::Integer>()) = std::move(k2);
   rel.insert("v", make_unique<algebra::BigInt>()) = std::move(v);
   rel.nrTuples = rows;

   auto t = Scan("t");
   HashGroup()
       .addKey(Column(t, "k1"), primitives::hash_int64_t_col,
               primitives::keys_not_equal_int64_t_col,
               primitives::partition_by_key_int64_t_col,
               primitives::scatter_sel_int64_t_col,
               primitives::keys_not_equal_row_int64_t_col,
               primitives::partition_by_key_row_int64_t_col,
               primitives::scatter_sel_row_int64_t_col,
               primitives::gather_val_int64_t_col,
               Buffer(grouped_k1, sizeof(int64_t)))
       .addKey(Column(t, "k2"), primitives::hash_int32_t_col,
               primitives::keys_not_equal_int32_t_col,
               primitives::partition_by_key_int32_t_col,
               primitives::scatter_sel_int32_t_col,
               primitives::keys_not_equal_row_int32_t_col,
               primitives::partition_by_key_row_int32_t_col,
               primitives::scatter_sel_row_int32_t_col,
               primitives::gather_val_int32_t_col,
               Buffer(grouped_k2, sizeof(int32_t)))
       .addValue(Column(t, "v"), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(aggregated_v, sizeof(int64_t)));

   auto root = popOperator();
   size_t found = 0;
   std::set<std::tuple<int64_t, int32_t>> seen;
   while (auto n = root->next()) {
      found += n;
      auto keys1 = (int64_t*)Buffer(grouped_k1).data;
      auto keys2 = (int32_t*)Buffer(grouped_k2).data;
      auto aggrs = (int64_t*)Buffer(aggregated_v).data;
      for (size_t i = 0; i < n; ++i) {
         const auto key = std::make_tuple(keys1[i], keys2[i]);
         ASSERT_TRUE(seen.insert(key).second) << "duplicate group";
         ASSERT_EQ(aggrs[i], expected.at(key));
      }
   }
   ASSERT_EQ(found, expected.size());
}

// Compound keys of the shapes VW_GROUP_DISPATCH_ROWKEY builds in a register
// (two 1-byte keys through one selection vector, TPC-H Q1; two 4-byte keys,
// dense or through one selection vector), with enough groups to create
// entries in several vectors and to share chains; every group's sum is
// checked. Without the option (or VW_GROUP_DISPATCH) they run the existing
// paths and must give the same groups.
namespace {
template <typename K> struct RowKeyPrims;
template <> struct RowKeyPrims<types::Char<1>> {
   static types::Char<1> make(uint64_t x) {
      types::Char<1> c;
      c.value = char('A' + x % 26);
      return c;
   }
   static std::unique_ptr<algebra::Type> type() {
      return make_unique<algebra::Char>(1);
   }
   static constexpr auto hashSel = &primitives::hash_sel_Char_1_col;
   static constexpr auto rehashSel = &primitives::rehash_sel_Char_1_col;
   static constexpr auto neqSel = &primitives::keys_not_equal_sel_Char_1_col;
   static constexpr auto partSel = &primitives::partition_by_key_sel_Char_1_col;
   static constexpr auto scatter = &primitives::scatter_sel_Char_1_col;
   static constexpr auto neqRow = &primitives::keys_not_equal_row_Char_1_col;
   static constexpr auto partRow = &primitives::partition_by_key_row_Char_1_col;
   static constexpr auto scatterRow = &primitives::scatter_sel_row_Char_1_col;
   static constexpr auto gather = &primitives::gather_val_Char_1_col;
};
template <> struct RowKeyPrims<int32_t> {
   static int32_t make(uint64_t x) { return int32_t(x % 33) - 7; }
   static std::unique_ptr<algebra::Type> type() {
      return make_unique<algebra::Integer>();
   }
   static constexpr auto hash = &primitives::hash_int32_t_col;
   static constexpr auto rehash = &primitives::rehash_int32_t_col;
   static constexpr auto neq = &primitives::keys_not_equal_int32_t_col;
   static constexpr auto part = &primitives::partition_by_key_int32_t_col;
   static constexpr auto hashSel = &primitives::hash_sel_int32_t_col;
   static constexpr auto rehashSel = &primitives::rehash_sel_int32_t_col;
   static constexpr auto neqSel = &primitives::keys_not_equal_sel_int32_t_col;
   static constexpr auto partSel = &primitives::partition_by_key_sel_int32_t_col;
   static constexpr auto scatter = &primitives::scatter_sel_int32_t_col;
   static constexpr auto neqRow = &primitives::keys_not_equal_row_int32_t_col;
   static constexpr auto partRow = &primitives::partition_by_key_row_int32_t_col;
   static constexpr auto scatterRow = &primitives::scatter_sel_row_int32_t_col;
   static constexpr auto gather = &primitives::gather_val_int32_t_col;
};
} // namespace

class HashGroupRowKey : public HashGroupT {
 protected:
   int64_t bound = 500;
   template <typename K> void groupCheck(bool withSel);
};

template <typename K> void HashGroupRowKey::groupCheck(bool withSel) {
   using P = RowKeyPrims<K>;
   enum { grouped_k1, grouped_k2, aggregated_v, selScat, vSel };
   constexpr size_t rows = 20000;
   std::vector<K> k1(rows), k2(rows);
   std::vector<int64_t> v(rows);
   auto keyOf = [](const K& a, const K& b) {
      int64_t x, y;
      if constexpr (std::is_same_v<K, int32_t>)
         x = a, y = b;
      else
         x = a.value, y = b.value;
      return std::make_tuple(x, y);
   };
   std::map<std::tuple<int64_t, int64_t>, int64_t> expected;
   std::mt19937_64 rng(11);
   for (size_t i = 0; i < rows; ++i) {
      k1[i] = P::make(rng());
      k2[i] = P::make(rng() % 31);
      v[i] = int64_t(rng() % 1000);
      if (!withSel || v[i] < bound) expected[keyOf(k1[i], k2[i])] += v[i];
   }
   auto& rel = db["t"];
   rel.insert("k1", P::type()) = std::move(k1);
   rel.insert("k2", P::type()) = std::move(k2);
   rel.insert("v", make_unique<algebra::BigInt>()) = std::move(v);
   rel.nrTuples = rows;

   auto t = Scan("t");
   if (withSel) {
      Select(Expression().addOp(primitives::sel_less_int64_t_col_int64_t_val,
                                  Buffer(vSel, sizeof(pos_t)),
                                  Column(t, "v"), Value(&bound)));
      HashGroup()
          .pushKeySelVec(Buffer(vSel), Buffer(selScat, sizeof(pos_t)))
          .addKey(Column(t, "k1"), Buffer(vSel), *P::hashSel, *P::neqSel,
                  *P::partSel, Buffer(selScat, sizeof(pos_t)), *P::scatter,
                  *P::neqRow, *P::partRow, *P::scatterRow, *P::gather,
                  Buffer(grouped_k1, sizeof(K)))
          .addKey(Column(t, "k2"), Buffer(vSel), *P::rehashSel,
                  *P::neqSel, *P::partSel, Buffer(selScat, sizeof(pos_t)),
                  *P::scatter, *P::neqRow, *P::partRow, *P::scatterRow,
                  *P::gather, Buffer(grouped_k2, sizeof(K)))
          .addValue(Column(t, "v"), Buffer(vSel),
                    primitives::aggr_init_plus_int64_t_col,
                    primitives::aggr_sel_plus_int64_t_col,
                    primitives::aggr_row_plus_int64_t_col,
                    primitives::gather_val_int64_t_col,
                    Buffer(aggregated_v, sizeof(int64_t)));
   } else {
      if constexpr (std::is_same_v<K, int32_t>)
         HashGroup()
             .addKey(Column(t, "k1"), *P::hash, *P::neq, *P::part,
                     *P::scatter, *P::neqRow, *P::partRow, *P::scatterRow,
                     *P::gather, Buffer(grouped_k1, sizeof(K)))
             .addKey(Column(t, "k2"), *P::rehash, *P::neq, *P::part,
                     *P::scatter, *P::neqRow, *P::partRow, *P::scatterRow,
                     *P::gather, Buffer(grouped_k2, sizeof(K)))
             .addValue(Column(t, "v"), primitives::aggr_init_plus_int64_t_col,
                       primitives::aggr_plus_int64_t_col,
                       primitives::aggr_row_plus_int64_t_col,
                       primitives::gather_val_int64_t_col,
                       Buffer(aggregated_v, sizeof(int64_t)));
   }

   auto root = popOperator();
   size_t found = 0;
   std::set<std::tuple<int64_t, int64_t>> seen;
   while (auto n = root->next()) {
      found += n;
      auto keys1 = (K*)Buffer(grouped_k1).data;
      auto keys2 = (K*)Buffer(grouped_k2).data;
      auto aggrs = (int64_t*)Buffer(aggregated_v).data;
      for (size_t i = 0; i < n; ++i) {
         const auto key = keyOf(keys1[i], keys2[i]);
         ASSERT_TRUE(seen.insert(key).second) << "duplicate group";
         ASSERT_EQ(aggrs[i], expected.at(key));
      }
   }
   ASSERT_EQ(found, expected.size());
}

TEST_F(HashGroupRowKey, char1Sel) { groupCheck<types::Char<1>>(true); }
TEST_F(HashGroupRowKey, int32Sel) { groupCheck<int32_t>(true); }
TEST_F(HashGroupRowKey, int32Dense) { groupCheck<int32_t>(false); }

class HashGroupSmallBuf : public ::testing::Test,
                          public Query,
                          public QueryBuilder {

 protected:
   runtime::Database db;
   runtime::GlobalPool pool;
   HashGroupSmallBuf() : Query(), QueryBuilder(db, shared, 2) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   };
};

TEST_F(HashGroupSmallBuf, simpleGroupSmallBuffers) {

   enum { grouped_k1, grouped_k2, aggregated_v };
   auto& rel = db["t"];
   rel.insert("k1", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 1, 1, 1, 1, 3, 4, 8};
   rel.insert("k2", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 2, 1, 1, 2, 9, 17, 4};
   rel.insert("v", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{4, 8, 16, 1, 2, 3, 17, 4};
   rel.nrTuples = 8;

   auto t = Scan("t");
   HashGroup()
       .addKey(Column(t, "k1"), primitives::hash_int64_t_col,
               primitives::keys_not_equal_int64_t_col,
               primitives::partition_by_key_int64_t_col,
               primitives::scatter_sel_int64_t_col,
               primitives::keys_not_equal_row_int64_t_col,
               primitives::partition_by_key_row_int64_t_col,
               primitives::scatter_sel_row_int64_t_col,
               primitives::gather_val_int64_t_col,
               Buffer(grouped_k1, sizeof(int64_t)))
       .addKey(Column(t, "k2"), primitives::rehash_int32_t_col,
               primitives::keys_not_equal_int32_t_col,
               primitives::partition_by_key_int32_t_col,
               primitives::scatter_sel_int32_t_col,
               primitives::keys_not_equal_row_int32_t_col,
               primitives::partition_by_key_row_int32_t_col,
               primitives::scatter_sel_row_int32_t_col,
               primitives::gather_val_int32_t_col,
               Buffer(grouped_k2, sizeof(int32_t)))
       .addValue(Column(t, "v"), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(aggregated_v, sizeof(int64_t)));
   std::map<std::tuple<int64_t, int32_t>, int64_t> expectedGroups = {
       {make_tuple(1, 1), 21},
       {make_tuple(1, 2), 10},
       {make_tuple(3, 9), 3},
       {make_tuple(4, 17), 17},
       {make_tuple(8, 4), 4}};

   auto root = popOperator();
   size_t found = 0;
   size_t n;
   while ((n = root->next()) != 0) {
      found += n;
      auto keys1 = (int64_t*)Buffer(grouped_k1).data;
      auto keys2 = (int32_t*)Buffer(grouped_k2).data;
      auto aggrs = (int64_t*)Buffer(aggregated_v).data;
      for (size_t i = 0; i < n; ++i) {
         int64_t ex = expectedGroups[std::make_tuple(keys1[i], keys2[i])];
         ASSERT_EQ(aggrs[i], ex);
      }
   }

   ASSERT_EQ(found, size_t(5));
}

TEST_F(HashGroupSmallBuf, groupWithSel) {
   enum { grouped_k1, grouped_k2, aggregated_v, selScat, vSel };
   auto& rel = db["t"];
   rel.insert("k1", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{1, 44, 1, 1, 44, 1, 44, 44, 44, 1, 3, 4, 8};
   rel.insert("k2", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 33, 2, 1, 22, 1, 99, 11, 11, 2, 9, 17, 4};
   rel.insert("v", make_unique<algebra::BigInt>()) =
       std::vector<int64_t>{4, 99, 8, 16, 88, 1, 33, 33, 22, 2, 3, 17, 4};
   rel.nrTuples = 13;
   int64_t upperBound = 20;

   auto t = Scan("t");
   Select(Expression().addOp(primitives::sel_less_int64_t_col_int64_t_val,
                             Buffer(vSel, sizeof(pos_t)), Column(t, "v"),
                             Value(&upperBound)));
   HashGroup()
       .pushKeySelVec(Buffer(vSel), Buffer(selScat, sizeof(pos_t)))
       .addKey(Column(t, "k1"), Buffer(vSel), primitives::hash_sel_int64_t_col,
               primitives::keys_not_equal_sel_int64_t_col,
               primitives::partition_by_key_sel_int64_t_col,
               Buffer(selScat, sizeof(pos_t)),
               primitives::scatter_sel_int64_t_col,
               primitives::keys_not_equal_row_int64_t_col,
               primitives::partition_by_key_row_int64_t_col,
               primitives::scatter_sel_row_int64_t_col,
               primitives::gather_val_int64_t_col,
               Buffer(grouped_k1, sizeof(int64_t)))
       .addKey(
           Column(t, "k2"), Buffer(vSel), primitives::rehash_sel_int32_t_col,
           primitives::keys_not_equal_sel_int32_t_col,
           primitives::partition_by_key_sel_int32_t_col,
           Buffer(selScat, sizeof(pos_t)), primitives::scatter_sel_int32_t_col,
           primitives::keys_not_equal_row_int32_t_col,
           primitives::partition_by_key_row_int32_t_col,
           primitives::scatter_sel_row_int32_t_col,
           primitives::gather_val_int32_t_col,
           Buffer(grouped_k2, sizeof(int32_t)))
       .addValue(Column(t, "v"), Buffer(vSel),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(aggregated_v, sizeof(int64_t)));
   std::map<std::tuple<int64_t, int32_t>, int64_t> expectedGroups = {
       {make_tuple(1, 1), 21},
       {make_tuple(1, 2), 10},
       {make_tuple(3, 9), 3},
       {make_tuple(4, 17), 17},
       {make_tuple(8, 4), 4}};

   auto root = popOperator();
   size_t found = 0;
   size_t n;
   while ((n = root->next()) != 0) {
      found += n;
      auto keys1 = (int64_t*)Buffer(grouped_k1).data;
      auto keys2 = (int32_t*)Buffer(grouped_k2).data;
      auto aggrs = (int64_t*)Buffer(aggregated_v).data;
      for (size_t i = 0; i < n; ++i) {
         int64_t ex = expectedGroups[std::make_tuple(keys1[i], keys2[i])];
         ASSERT_EQ(aggrs[i], ex);
      }
   }

   ASSERT_EQ(found, size_t(5));
}

// A join the plan derives as semi (unique build key by declared primary key,
// no build column): each probe row whose key is in the build side is emitted
// once. With VW_JOIN_SEMI it runs joinSemi (or the bitmap under
// VW_JOIN_SEMI_BITMAP), without it the inner join path; same rows.
struct SemiJoinBuilder : public Query, public vectorwise::QueryBuilder {
   enum { probe_matches };
   runtime::GlobalPool pool;
   SemiJoinBuilder(runtime::Database& db) : Query(), QueryBuilder(db, shared) {
      previous = runtime::this_worker->allocator.setSource(&pool);
   }
   unique_ptr<vectorwise::Operator> getQuery() {
      auto build = Scan("build");
      auto probe = Scan("probe");
      HashJoin(Buffer(probe_matches, sizeof(pos_t)))
          .semi()
          .addBuildKey(Column(build, "k"), primitives::hash_int32_t_col,
                       primitives::scatter_int32_t_col)
          .addProbeKey(Column(probe, "b"), primitives::hash_int32_t_col,
                       primitives::keys_equal_int32_t_col);
      return popOperator();
   }
};

namespace {
/// every key hashes to the same value: one chain, every entry a hash match
pos_t hashConstant(pos_t n, void* result, void*) {
   auto* h = static_cast<runtime::Hashmap::hash_t*>(result);
   for (pos_t i = 0; i < n; ++i) h[i] = 0x5bd1e995ull << 20;
   return n;
}

void semiJoinCheck(bool collide) {
   runtime::Database db;
   db["build"].insert("k", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{1, 3, 4, 8};
   db["build"].primaryKey = {"k"};
   db["probe"].insert("b", make_unique<algebra::Integer>()) =
       std::vector<int32_t>{88, 1, 1, 17, 4, 3, 3, 8, 2};
   db["build"].nrTuples = 4;
   db["probe"].nrTuples = 9;

   SemiJoinBuilder b(db);
   auto root = b.getQuery();
   auto join = dynamic_cast<Hashjoin*>(root.get());
   ASSERT_NE(nullptr, join);
   ASSERT_TRUE(join->semiJoin);
   if (collide) {
      // the probe must compare keys along a chain of equal hashes
      static_cast<F2_Op*>(join->buildHash.ops[0].get())->operation = hashConstant;
      static_cast<F2_Op*>(join->probeHash.ops[0].get())->operation = hashConstant;
#ifdef VW_JOIN_FUSED_PROBE
      join->fusedOk = false; // fused probes hash the keys themselves
#endif
   }
   std::vector<pos_t> got;
   for (auto n = root->next(); n != EndOfStream; n = root->next())
      got.insert(got.end(), join->probeMatches, join->probeMatches + n);
   std::sort(got.begin(), got.end());
   EXPECT_EQ((std::vector<pos_t>{1, 2, 4, 5, 6, 7}), got);
#ifdef VW_JOIN_SEMI
#ifdef VW_JOIN_SEMI_BITMAP
   EXPECT_NE(join->semiHash, join->semiActive);
#else
   EXPECT_TRUE(join->semiHash);
#endif
#endif
}
} // namespace

TEST(Join, semiJoin) { semiJoinCheck(false); }
TEST(Join, semiJoinHashCollisions) { semiJoinCheck(true); }

} // namespace operatortest
