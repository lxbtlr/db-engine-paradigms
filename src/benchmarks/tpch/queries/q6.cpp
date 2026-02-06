#include "benchmarks/tpch/Queries.hpp"
#include "common/runtime/Types.hpp"
#include "tbb/tbb.h"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Operators.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include "vectorwise/VectorAllocator.hpp"
#include <iostream>

using namespace runtime;
using namespace std;

static constexpr bool DEBUG_Q6 = true;

NOVECTORIZE Relation q6_hyper(Database& db, size_t /*nrThreads*/) {
   Relation result;
   result.insert("revenue", make_unique<algebra::Numeric>(12, 4));

   auto& rel = db["lineitem"];
   
   auto* __restrict__ l_shipdate_raw = reinterpret_cast<const int32_t*>(rel["l_shipdate"].data<types::Date>());
   auto* __restrict__ l_quantity_raw = reinterpret_cast<const int64_t*>(rel["l_quantity"].data<types::Numeric<12, 2>>());
   auto* __restrict__ l_price_raw    = reinterpret_cast<const int64_t*>(rel["l_extendedprice"].data<types::Numeric<12, 2>>());
   auto* __restrict__ l_discount_raw = reinterpret_cast<const int64_t*>(rel["l_discount"].data<types::Numeric<12, 2>>());
   
   int32_t c1_v = types::Date::castString("1994-01-01").value;
   int32_t c2_v = types::Date::castString("1995-01-01").value;
   int64_t c3_v = types::Numeric<12, 2>::castString("0.05").value;
   int64_t c4_v = types::Numeric<12, 2>::castString("0.07").value;
   int64_t c5_v = 24*100; // quantity < 24.00

   types::Numeric<12, 4> total_revenue = tbb::parallel_reduce(
       tbb::blocked_range<size_t>(0, rel.nrTuples),
       types::Numeric<12, 4>(0),
       [=](const tbb::blocked_range<size_t>& r, types::Numeric<12, 4> revenue) {
           int64_t local_acc = revenue.value;
		
	   size_t start = r.begin();
	   size_t end = r.end();
	   
	   const int32_t* __restrict__ s_date = l_shipdate_raw;
           const int64_t* __restrict__ s_quant = l_quantity_raw;
           const int64_t* __restrict__ s_price = l_price_raw;
           const int64_t* __restrict__ s_disc  = l_discount_raw;

	   for (size_t i = start; i < end; ++i) {
               if (s_date[i] >= c1_v && s_date[i] < c2_v) {
                  if (s_disc[i] >= c3_v && s_disc[i] <= c4_v) {
                     if (s_quant[i] < c5_v) {
                        local_acc += (s_price[i] * s_disc[i]);
                     }
                  }
               }
           }
        return types::Numeric<12, 4>::buildRaw(local_acc);
       },
       [](types::Numeric<12, 4> x, types::Numeric<12, 4> y) {
           return types::Numeric<12, 4>::buildRaw(x.value + y.value);
       }
   );

   // monitor correctness (debug)
   if (DEBUG_Q6) {
       // We divide by 10000.0 because Numeric<12,4> represents 4 decimal places
       double final_val = static_cast<double>(total_revenue.value) / 10000.0;
       fprintf(stderr, "\n--- TPC-H Q6 Monitor ---\n");
       fprintf(stderr, "Total Tuples Scanned: %zu\n", rel.nrTuples);
       fprintf(stderr, "Raw Acc (Scale 1):    %ld\n", total_revenue.value);
       fprintf(stderr, "Final Revenue:        %.4f\n", final_val);
       fprintf(stderr, "Target Result:   123141147.1752\n");
       fprintf(stderr, "------------------------\n\n");
   }

   auto& rev_col = result["revenue"].typedAccessForChange<types::Numeric<12, 4>>();
   rev_col.reset(1);
   rev_col.push_back(total_revenue);
   result.nrTuples = 1;

   return result;
}

unique_ptr<Q6Builder::Q6> Q6Builder::getQuery() {
   using namespace vectorwise;
   // --- constants
   auto res = make_unique<Q6>();
   auto& consts = *res;
   enum { sel_a, sel_b, result_project };

   assert(db["lineitem"]["l_shipdate"].type->rt_size() == sizeof(consts.c2));
   assert(db["lineitem"]["l_quantity"].type->rt_size() == sizeof(consts.c5));
   assert(db["lineitem"]["l_discount"].type->rt_size() == sizeof(consts.c3));
   assert(db["lineitem"]["l_extendedprice"].type->rt_size() == sizeof(int64_t));

   auto lineitem = Scan("lineitem");
   Select((Expression()                                       //
              .addOp(conf.sel_less_int32_t_col_int32_t_val(), //
                     Buffer(sel_a, sizeof(pos_t)),            //
                     Column(lineitem, "l_shipdate"),          //
                     Value(&consts.c2)))
              .addOp(conf.selsel_greater_equal_int32_t_col_int32_t_val(), //
                     Buffer(sel_a, sizeof(pos_t)),                        //
                     Buffer(sel_b, sizeof(pos_t)),                        //
                     Column(lineitem, "l_shipdate"),                      //
                     Value(&consts.c1))
             //.addOp(conf.selsel_less_int64_t_col_int64_t_val(), //
             .addOp(primitives::selsel_less_int64_t_col_int64_t_val_bf, //
                     Buffer(sel_b, sizeof(pos_t)),               //
                     Buffer(sel_a, sizeof(pos_t)),               //
                     Column(lineitem, "l_quantity"),             //
                     Value(&consts.c5))
              //.addOp(conf.selsel_greater_equal_int64_t_col_int64_t_val(), //
              .addOp(primitives::selsel_greater_equal_int64_t_col_int64_t_val_bf, //
                     Buffer(sel_a, sizeof(pos_t)),                        //
                     Buffer(sel_b, sizeof(pos_t)),                        //
                     Column(lineitem, "l_discount"),                      //
                     Value(&consts.c3))
              .addOp(conf.selsel_less_equal_int64_t_col_int64_t_val(), //
                     Buffer(sel_b, sizeof(pos_t)),                     //
                     Buffer(sel_a, sizeof(pos_t)),                     //
                     Column(lineitem, "l_discount"),                   //
                     Value(&consts.c4)));
   Project().addExpression(
       Expression() //
           .addOp(primitives::proj_sel_both_multiplies_int64_t_col_int64_t_col,
                  Buffer(sel_a),                           //
                  Buffer(result_project, sizeof(int64_t)), //
                  Column(lineitem, "l_discount"),
                  Column(lineitem, "l_extendedprice")));
   FixedAggregation(Expression() //
                        .addOp(primitives::aggr_static_plus_int64_t_col,
                               Value(&consts.aggregator), //
                               Buffer(result_project)));
   res->rootOp = popOperator();
   assert(operatorStack.size() == 0);
   return res;
}

Relation q6_vectorwise(Database& db, size_t nrThreads, size_t vectorSize) {
   using namespace vectorwise;

   std::atomic<size_t> n;
   runtime::Relation result;
   vectorwise::SharedStateManager shared;
   WorkerGroup workers(nrThreads);
   GlobalPool pool;
   std::atomic<int64_t> aggr;
   aggr = 0;
   n = 0;
   workers.run([&]() {
      Q6Builder b(db, shared, vectorSize);
      b.previous = this_worker->allocator.setSource(&pool);
      auto query = b.getQuery();
      auto n_ = query->rootOp->next();
      if (n_) {
         aggr.fetch_add(query->aggregator);
         n.fetch_add(n_);
      }

      auto leader = barrier();
      if (leader) {
         result.insert("revenue", make_unique<algebra::Numeric>(12, 4));
         if (n.load()) {
            auto& sum =
                result["revenue"].template typedAccessForChange<int64_t>();
            sum.reset(1);
            auto a = aggr.load();
            sum.push_back(a);
            result.nrTuples = 1;
         }
      }
   });

   return result;
}
