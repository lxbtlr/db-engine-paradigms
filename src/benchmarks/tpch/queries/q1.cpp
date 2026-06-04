#include "benchmarks/tpch/Queries.hpp"
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "hyper/GroupBy.hpp"
#include "hyper/ParallelHelper.hpp"
#include "tbb/tbb.h"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Operators.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include "vectorwise/VectorAllocator.hpp"
#include <deque>
#include <iostream>

using namespace runtime;
using namespace std;
using vectorwise::primitives::Char_1;
using vectorwise::primitives::hash_t;


//  select
//    l_returnflag,
//    l_linestatus,
//    sum(l_quantity) as sum_qty,
//    sum(l_extendedprice) as sum_base_price,
//    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
//    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
//    avg(l_quantity) as avg_qty,
//    avg(l_extendedprice) as avg_price,
//    avg(l_discount) as avg_disc,
//    count(*) as count_order
//  from
//    lineitem
//  where
//    l_shipdate <= date '1998-12-01' - interval '90' day
//  group by
//    l_returnflag,
//    l_linestatus
//
//  NOTE: extra aggregates (get<5> through get<12>) are deliberate duplicates
//  of the original aggregates, added to stress register pressure in Hyper.
//  Anti-CSE strategy: multiply inputs by runtime constants that are always 1
//  but are not visible to the optimizer at compile time, forcing separate
//  codegen for each duplicate set.
//  If Hyper's stores/tuple climbs toward Vectorwise's as these are added,
//  that confirms the register residency claim. If both stay flat or climb
//  proportionally, it challenges it.

// Runtime constants used to defeat CSE. Marked volatile so the compiler
// cannot assume their value at compile time and cannot fold duplicate
// expressions back into a single computation.

#ifdef STRESS_TEST

static volatile int64_t cse_barrier_1 = 1;
static volatile int64_t cse_barrier_2 = 1;
static volatile int64_t cse_barrier_3 = 1;

NOVECTORIZE std::unique_ptr<runtime::Query> q1_hyper(Database& db,
                                                     size_t nrThreads) {
   using namespace types;
   using namespace std;
   types::Date c1 = types::Date::castString("1998-09-02");
   types::Numeric<12, 2> one = types::Numeric<12, 2>::castString("1.00");
   auto& li = db["lineitem"];
   auto l_returnflag = li["l_returnflag"].data<types::Char<1>>();
   auto l_linestatus = li["l_linestatus"].data<types::Char<1>>();
   auto l_extendedprice = li["l_extendedprice"].data<types::Numeric<12, 2>>();
   auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
   auto l_tax = li["l_tax"].data<types::Numeric<12, 2>>();
   auto l_quantity = li["l_quantity"].data<types::Numeric<12, 2>>();
   auto l_shipdate = li["l_shipdate"].data<types::Date>();

   auto resources = initQuery(nrThreads);

   using hash = runtime::CRC32Hash;

   // Read the volatile barriers once before the parallel region so the
   // value is in scope for the lambda but still opaque to the optimizer.
   // These will always be 1 at runtime; the compiler cannot know that.
   const int64_t b1 = cse_barrier_1;
   const int64_t b2 = cse_barrier_2;
   const int64_t b3 = cse_barrier_3;

   // 20 accumulators: 5 original + 3 duplicate sets of 5 (inc. count_order).
   // get<0>  = sum_qty          (original)
   // get<1>  = sum_base_price   (original)
   // get<2>  = sum_disc_price   (original)
   // get<3>  = sum_charge       (original)
   // get<4>  = count_order      (original)
   // get<5>  = sum_qty_2        (duplicate set 1, barrier b1)
   // get<6>  = sum_base_price_2 (duplicate set 1)
   // get<7>  = sum_disc_price_2 (duplicate set 1)
   // get<8>  = sum_charge_2     (duplicate set 1)
   // get<9>  = count_order_2    (duplicate set 1)
   // get<10> = sum_qty_3        (duplicate set 2, barrier b2)
   // get<11> = sum_base_price_3 (duplicate set 2)
   // get<12> = sum_disc_price_3 (duplicate set 2)
   // get<13> = sum_charge_3     (duplicate set 2)
   // get<14> = count_order_3    (duplicate set 2)
   // get<15> = sum_qty_4        (duplicate set 3, barrier b3)
   // get<16> = sum_base_price_4 (duplicate set 3)
   // get<17> = sum_disc_price_4 (duplicate set 3)
   // get<18> = sum_charge_4     (duplicate set 3)
   // get<19> = count_order_4    (duplicate set 3)
   auto groupOp = make_GroupBy<tuple<Char<1>, Char<1>>,
                               tuple<Numeric<12, 2>, Numeric<12, 2>,
                                     Numeric<12, 4>, Numeric<12, 6>, int64_t,
                                     Numeric<12, 2>, Numeric<12, 2>,
                                     Numeric<12, 4>, Numeric<12, 6>, int64_t,
                                     Numeric<12, 2>, Numeric<12, 2>,
                                     Numeric<12, 4>, Numeric<12, 6>, int64_t,
                                     Numeric<12, 2>, Numeric<12, 2>,
                                     Numeric<12, 4>, Numeric<12, 6>, int64_t>,
                               hash>(
       [](auto& acc, auto&& value) {
          // original aggregates
          get<0>(acc) += get<0>(value);
          get<1>(acc) += get<1>(value);
          get<2>(acc) += get<2>(value);
          get<3>(acc) += get<3>(value);
          get<4>(acc) += get<4>(value);
          // duplicate set 1
          get<5>(acc)  += get<5>(value);
          get<6>(acc)  += get<6>(value);
          get<7>(acc)  += get<7>(value);
          get<8>(acc)  += get<8>(value);
          get<9>(acc)  += get<9>(value);
          // duplicate set 2
          get<10>(acc) += get<10>(value);
          get<11>(acc) += get<11>(value);
          get<12>(acc) += get<12>(value);
          get<13>(acc) += get<13>(value);
          get<14>(acc) += get<14>(value);
          // duplicate set 3
          get<15>(acc) += get<15>(value);
          get<16>(acc) += get<16>(value);
          get<17>(acc) += get<17>(value);
          get<18>(acc) += get<18>(value);
          get<19>(acc) += get<19>(value);
       },
       make_tuple(Numeric<12, 2>(), Numeric<12, 2>(), Numeric<12, 4>(),
                  Numeric<12, 6>(), int64_t(0),
                  Numeric<12, 2>(), Numeric<12, 2>(), Numeric<12, 4>(),
                  Numeric<12, 6>(), int64_t(0),
                  Numeric<12, 2>(), Numeric<12, 2>(), Numeric<12, 4>(),
                  Numeric<12, 6>(), int64_t(0),
                  Numeric<12, 2>(), Numeric<12, 2>(), Numeric<12, 4>(),
                  Numeric<12, 6>(), int64_t(0)),
       nrThreads);

   tbb::parallel_for(
       tbb::blocked_range<size_t>(0, li.nrTuples, morselSize),
       [&](const tbb::blocked_range<size_t>& r) {
          auto locals = groupOp.preAggLocals();
          for (size_t i = r.begin(), end = r.end(); i != end; ++i) {
             if (l_shipdate[i] <= c1) {
                auto& group =
                    locals.getGroup(make_tuple(l_returnflag[i], l_linestatus[i]));

                // Anti-pattern: deferred consumption + binary tree dependencies
                // + cross-set dependencies to force register spilling.
                //
                // All 12 intermediates are computed before any accumulator is
                // written. Cross-set multiplications create binary tree nodes
                // that require values from multiple sets to be simultaneously
                // live. The wide fan-in at the end forces ~12 values live at
                // the point of accumulation.

                // --- compute all intermediates up front (deferred consumption)
                // set 0 (original inputs)
                auto disc_price  = l_extendedprice[i] * (one - l_discount[i]);
                auto charge      = disc_price * (one + l_tax[i]);
                auto qty0        = l_quantity[i];

                // set 1 (barrier b1) -- kept live, not yet consumed
                auto ep1         = Numeric<12,2>(l_extendedprice[i].value * b1);
                auto disc1       = Numeric<12,2>(l_discount[i].value * b1);
                auto tax1        = Numeric<12,2>(l_tax[i].value * b1);
                auto disc_price2 = ep1 * (one - disc1);
                auto charge2     = disc_price2 * (one + tax1);

                // set 2 (barrier b2) -- kept live, not yet consumed
                auto ep2         = Numeric<12,2>(l_extendedprice[i].value * b2);
                auto disc2       = Numeric<12,2>(l_discount[i].value * b2);
                auto tax2        = Numeric<12,2>(l_tax[i].value * b2);
                auto disc_price3 = ep2 * (one - disc2);
                auto charge3     = disc_price3 * (one + tax2);

                // set 3 (barrier b3) -- kept live, not yet consumed
                auto ep3         = Numeric<12,2>(l_extendedprice[i].value * b3);
                auto disc3       = Numeric<12,2>(l_discount[i].value * b3);
                auto tax3        = Numeric<12,2>(l_tax[i].value * b3);
                auto disc_price4 = ep3 * (one - disc3);
                auto charge4     = disc_price4 * (one + tax3);

                // --- binary tree cross-set combinations
                // Each node requires two intermediates from different sets to
                // be simultaneously live, widening the live set further.
                auto cross_dp_01 = disc_price  * disc_price2; // sets 0+1 live
                auto cross_dp_23 = disc_price3 * disc_price4; // sets 2+3 live
                auto cross_ch_01 = charge       * charge2;    // sets 0+1 live
                auto cross_ch_23 = charge3      * charge4;    // sets 2+3 live
                // tree depth 2: all four disc_price values must still be live
                auto tree_dp     = cross_dp_01 * cross_dp_23;
                auto tree_ch     = cross_ch_01 * cross_ch_23;

                // --- wide fan-in: all intermediates simultaneously live at
                // the point of accumulation (anti-pattern 4)
                get<0>(group)  += qty0;
                get<1>(group)  += disc_price;
                get<2>(group)  += charge;
                get<3>(group)  += disc_price2;
                get<4>(group)  += 1;
                get<5>(group)  += charge2;
                get<6>(group)  += ep1;
                get<7>(group)  += disc_price3;
                get<8>(group)  += charge3;
                get<9>(group)  += 1;
                get<10>(group) += ep2;
                get<11>(group) += disc_price4;
                get<12>(group) += charge4;
                get<13>(group) += ep3;
                get<14>(group) += 1;
                // tree nodes written last — forces tree values live across all
                // prior accumulator writes above (cross-set anti-pattern)
                get<15>(group) += tree_dp;
                get<16>(group) += tree_ch;
                get<17>(group) += cross_dp_01;
                get<18>(group) += cross_ch_01;
                get<19>(group) += 1;
             }
          }
       });

   auto& result = resources.query->result;
   auto retAttr = result->addAttribute("l_returnflag", sizeof(Char<1>));
   auto statusAttr = result->addAttribute("l_linestatus", sizeof(Char<1>));
   auto qtyAttr = result->addAttribute("sum_qty", sizeof(Numeric<12, 2>));
   auto base_priceAttr =
       result->addAttribute("sum_base_price", sizeof(Numeric<12, 2>));
   auto disc_priceAttr =
       result->addAttribute("sum_disc_price", sizeof(Numeric<12, 4>));
   auto chargeAttr = result->addAttribute("sum_charge", sizeof(Numeric<12, 6>));
   auto count_orderAttr = result->addAttribute("count_order", sizeof(int64_t));

   groupOp.forallGroups([&](runtime::Stack<decltype(groupOp)::group_t>& entries) {
      auto n = entries.size();
      auto block = result->createBlock(n);
      auto ret = reinterpret_cast<Char<1>*>(block.data(retAttr));
      auto status = reinterpret_cast<Char<1>*>(block.data(statusAttr));
      auto qty = reinterpret_cast<Numeric<12, 2>*>(block.data(qtyAttr));
      auto base_price =
          reinterpret_cast<Numeric<12, 2>*>(block.data(base_priceAttr));
      auto disc_price =
          reinterpret_cast<Numeric<12, 4>*>(block.data(disc_priceAttr));
      auto charge = reinterpret_cast<Numeric<12, 6>*>(block.data(chargeAttr));
      auto count_order =
          reinterpret_cast<int64_t*>(block.data(count_orderAttr));
      for (auto block : entries)
         for (auto& entry : block) {
            *ret++ = get<0>(entry.k);
            *status++ = get<1>(entry.k);
            *qty++ = get<0>(entry.v);
            *base_price++ = get<1>(entry.v);
            *disc_price++ = get<2>(entry.v);
            *charge++ = get<3>(entry.v);
            *count_order++ = get<4>(entry.v);
            // extra aggregates computed but not written to result output --
            // they exist purely to force register pressure in the hot loop.
            // They are retained in the group table so the compiler cannot
            // dead-code-eliminate the inner loop computations.
         }
      block.addedElements(n);
   });

   leaveQuery(nrThreads);
   return move(resources.query);
}

std::unique_ptr<Q1Builder::Q1> Q1Builder::getQuery() {
   using namespace vectorwise;
   auto result = Result();
   previous = result.resultWriter.shared.result->participate();

   auto r = make_unique<Q1>();
   auto lineitem = Scan("lineitem");
   Select(Expression().addOp(BF(primitives::sel_less_equal_Date_col_Date_val),
                             Buffer(sel_date, sizeof(pos_t)),
                             Column(lineitem, "l_shipdate"), Value(&r->c1)));
   Project()
       .addExpression(
           Expression()
               .addOp(conf.proj_sel_minus_int64_t_val_int64_t_col(),
                      Buffer(sel_date),
                      Buffer(result_proj_minus, sizeof(int64_t)),
                      Value(&r->one), Column(lineitem, "l_discount"))
               .addOp(conf.proj_multiplies_sel_int64_t_col_int64_t_col(),
                      Buffer(sel_date), Buffer(disc_price, sizeof(int64_t)),
                      Column(lineitem, "l_extendedprice"),
                      Buffer(result_proj_minus, sizeof(int64_t))))
       .addExpression(
           Expression()
               .addOp(conf.proj_sel_plus_int64_t_col_int64_t_val(),
                      Buffer(sel_date),
                      Buffer(result_proj_plus, sizeof(int64_t)),
                      Column(lineitem, "l_tax"), Value(&r->one))
               .addOp(conf.proj_multiplies_int64_t_col_int64_t_col(),
                      Buffer(charge, sizeof(int64_t)),
                      Buffer(disc_price, sizeof(int64_t)),
                      Buffer(result_proj_plus, sizeof(int64_t))));
   HashGroup()
       .pushKeySelVec(Buffer(sel_date), Buffer(sel_date_grouped, sizeof(pos_t)))
       .addKey(Column(lineitem, "l_returnflag"), Buffer(sel_date),
               primitives::hash_sel_Char_1_col,
               primitives::keys_not_equal_sel_Char_1_col,
               primitives::partition_by_key_sel_Char_1_col,
               Buffer(sel_date_grouped, sizeof(pos_t)),
               primitives::scatter_sel_Char_1_col,
               primitives::keys_not_equal_row_Char_1_col,
               primitives::partition_by_key_row_Char_1_col,
               primitives::scatter_sel_row_Char_1_col,
               primitives::gather_val_Char_1_col,
               Buffer(returnflag, sizeof(Char_1)))
       .addKey(Column(lineitem, "l_linestatus"), Buffer(sel_date),
               primitives::rehash_sel_Char_1_col,
               primitives::keys_not_equal_sel_Char_1_col,
               primitives::partition_by_key_sel_Char_1_col,
               Buffer(sel_date_grouped, sizeof(pos_t)),
               primitives::scatter_sel_Char_1_col,
               primitives::keys_not_equal_row_Char_1_col,
               primitives::partition_by_key_row_Char_1_col,
               primitives::scatter_sel_row_Char_1_col,
               primitives::gather_val_Char_1_col,
               Buffer(linestatus, sizeof(Char_1)))
       .padToAlign(sizeof(types::Numeric<12, 4>))
       // original aggregates
       .addValue(Buffer(disc_price), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_disc_price, sizeof(types::Numeric<12, 4>)))
       .addValue(Buffer(charge), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_charge, sizeof(types::Numeric<12, 4>)))
       .addValue(Column(lineitem, "l_quantity"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_qty, sizeof(types::Numeric<12, 2>)))
       .addValue(Column(lineitem, "l_extendedprice"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_base_price, sizeof(types::Numeric<12, 2>)))
       .addValue(Buffer(charge, sizeof(uint64_t)),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_count_star,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(count_order, sizeof(uint64_t)))
       // duplicate set 1
       .addValue(Buffer(disc_price), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_disc_price_2, sizeof(types::Numeric<12, 4>)))
       .addValue(Buffer(charge), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_charge_2, sizeof(types::Numeric<12, 4>)))
       .addValue(Column(lineitem, "l_quantity"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_qty_2, sizeof(types::Numeric<12, 2>)))
       .addValue(Column(lineitem, "l_extendedprice"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_base_price_2, sizeof(types::Numeric<12, 2>)))
       .addValue(Buffer(charge, sizeof(uint64_t)),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_count_star,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(count_order_2, sizeof(uint64_t)))
       // duplicate set 2
       .addValue(Buffer(disc_price), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_disc_price_3, sizeof(types::Numeric<12, 4>)))
       .addValue(Buffer(charge), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_charge_3, sizeof(types::Numeric<12, 4>)))
       .addValue(Column(lineitem, "l_quantity"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_qty_3, sizeof(types::Numeric<12, 2>)))
       .addValue(Column(lineitem, "l_extendedprice"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_base_price_3, sizeof(types::Numeric<12, 2>)))
       .addValue(Buffer(charge, sizeof(uint64_t)),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_count_star,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(count_order_3, sizeof(uint64_t)))
       // duplicate set 3
       .addValue(Buffer(disc_price), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_disc_price_4, sizeof(types::Numeric<12, 4>)))
       .addValue(Buffer(charge), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_charge_4, sizeof(types::Numeric<12, 4>)))
       .addValue(Column(lineitem, "l_quantity"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_qty_4, sizeof(types::Numeric<12, 2>)))
       .addValue(Column(lineitem, "l_extendedprice"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_base_price_4, sizeof(types::Numeric<12, 2>)))
       .addValue(Buffer(charge, sizeof(uint64_t)),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_count_star,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(count_order_4, sizeof(uint64_t)));

   result.addValue("l_returnflag", Buffer(returnflag))
       .addValue("l_linestatus", Buffer(linestatus))
       .addValue("sum_qty", Buffer(sum_qty))
       .addValue("sum_base_price", Buffer(sum_base_price))
       .addValue("sum_disc_price", Buffer(sum_disc_price))
       .addValue("sum_charge", Buffer(sum_charge))
       .addValue("count_order", Buffer(count_order))
       .addValue("sum_qty_2", Buffer(sum_qty_2))
       .addValue("sum_base_price_2", Buffer(sum_base_price_2))
       .addValue("sum_disc_price_2", Buffer(sum_disc_price_2))
       .addValue("sum_charge_2", Buffer(sum_charge_2))
       .addValue("count_order_2", Buffer(count_order_2))
       .addValue("sum_qty_3", Buffer(sum_qty_3))
       .addValue("sum_base_price_3", Buffer(sum_base_price_3))
       .addValue("sum_disc_price_3", Buffer(sum_disc_price_3))
       .addValue("sum_charge_3", Buffer(sum_charge_3))
       .addValue("count_order_3", Buffer(count_order_3))
       .addValue("sum_qty_4", Buffer(sum_qty_4))
       .addValue("sum_base_price_4", Buffer(sum_base_price_4))
       .addValue("sum_disc_price_4", Buffer(sum_disc_price_4))
       .addValue("sum_charge_4", Buffer(sum_charge_4))
       .addValue("count_order_4", Buffer(count_order_4))
       .finalize();

   r->rootOp = popOperator();
   return r;
}

std::unique_ptr<runtime::Query> q1_vectorwise(Database& db, size_t nrThreads,
                                              size_t vectorSize) {
   using namespace vectorwise;
   WorkerGroup workers(nrThreads);
   vectorwise::SharedStateManager shared;

   std::unique_ptr<runtime::Query> result;
   workers.run([&]() {
      Q1Builder builder(db, shared, vectorSize);
      auto query = builder.getQuery();
      /* auto found = */ query->rootOp->next();
      auto leader = barrier();
      if (leader)
         result = move(
             dynamic_cast<ResultWriter*>(query->rootOp.get())->shared.result);
   });

   return result;
}

#endif

#ifndef STRESS_TEST


NOVECTORIZE std::unique_ptr<runtime::Query> q1_hyper(Database& db,
                                                     size_t nrThreads) {
   using namespace types;
   using namespace std;
   types::Date c1 = types::Date::castString("1998-09-02");
   types::Numeric<12, 2> one = types::Numeric<12, 2>::castString("1.00");
   auto& li = db["lineitem"];
   auto l_returnflag = li["l_returnflag"].data<types::Char<1>>();
   auto l_linestatus = li["l_linestatus"].data<types::Char<1>>();
   auto l_extendedprice = li["l_extendedprice"].data<types::Numeric<12, 2>>();
   auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
   auto l_tax = li["l_tax"].data<types::Numeric<12, 2>>();
   auto l_quantity = li["l_quantity"].data<types::Numeric<12, 2>>();
   auto l_shipdate = li["l_shipdate"].data<types::Date>();

   auto resources = initQuery(nrThreads);

   using hash = runtime::CRC32Hash;

   auto groupOp = make_GroupBy<tuple<Char<1>, Char<1>>,
                               tuple<Numeric<12, 2>, Numeric<12, 2>,
                                     Numeric<12, 4>, Numeric<12, 6>, int64_t>,
                               hash>(
       [](auto& acc, auto&& value) {
          get<0>(acc) += get<0>(value);
          get<1>(acc) += get<1>(value);
          get<2>(acc) += get<2>(value);
          get<3>(acc) += get<3>(value);
          get<4>(acc) += get<4>(value);
       },
       make_tuple(Numeric<12, 2>(), Numeric<12, 2>(), Numeric<12, 4>(),
                  Numeric<12, 6>(), int64_t(0)),
       nrThreads);


   tbb::parallel_for(
       tbb::blocked_range<size_t>(0, li.nrTuples, morselSize),
       [&](const tbb::blocked_range<size_t>& r) {
          auto locals = groupOp.preAggLocals();
          for (size_t i = r.begin(), end = r.end(); i != end; ++i) {
             if (l_shipdate[i] <= c1) {
                auto& group = locals.getGroup(make_tuple(l_returnflag[i], l_linestatus[i]));

                get<0>(group) += l_quantity[i];
                get<1>(group) += l_extendedprice[i];
                auto disc_price = l_extendedprice[i] * (one - l_discount[i]);
                get<2>(group) += disc_price;
                auto charge = disc_price * (one + l_tax[i]);
                get<3>(group) += charge;
                get<4>(group) += 1;
             }
          }
       });

   auto& result = resources.query->result;
   auto retAttr = result->addAttribute("l_returnflag", sizeof(Char<1>));
   auto statusAttr = result->addAttribute("l_linestatus", sizeof(Char<1>));
   auto qtyAttr = result->addAttribute("sum_qty", sizeof(Numeric<12, 2>));
   auto base_priceAttr =
       result->addAttribute("sum_base_price", sizeof(Numeric<12, 2>));
   auto disc_priceAttr =
       result->addAttribute("sum_disc_price", sizeof(Numeric<12, 2>));
   auto chargeAttr = result->addAttribute("sum_charge", sizeof(Numeric<12, 2>));
   auto count_orderAttr = result->addAttribute("count_order", sizeof(int64_t));

   groupOp.forallGroups([&](runtime::Stack<decltype(groupOp)::group_t>&  entries) {
      auto n = entries.size();
      auto block = result->createBlock(n);
      auto ret = reinterpret_cast<Char<1>*>(block.data(retAttr));
      auto status = reinterpret_cast<Char<1>*>(block.data(statusAttr));
      auto qty = reinterpret_cast<Numeric<12, 2>*>(block.data(qtyAttr));
      auto base_price =
          reinterpret_cast<Numeric<12, 2>*>(block.data(base_priceAttr));
      auto disc_price =
          reinterpret_cast<Numeric<12, 4>*>(block.data(disc_priceAttr));
      auto charge = reinterpret_cast<Numeric<12, 6>*>(block.data(chargeAttr));
      auto count_order =
          reinterpret_cast<int64_t*>(block.data(count_orderAttr));
      for (auto block : entries)
         for (auto& entry : block) {
            *ret++ = get<0>(entry.k);
            *status++ = get<1>(entry.k);
            *qty++ = get<0>(entry.v);
            *base_price++ = get<1>(entry.v);
            *disc_price++ = get<2>(entry.v);
            *charge++ = get<3>(entry.v);
            *count_order++ = get<4>(entry.v);
         }
      block.addedElements(n);
   });

   leaveQuery(nrThreads);
   return move(resources.query);
}

std::unique_ptr<Q1Builder::Q1> Q1Builder::getQuery() {
   using namespace vectorwise;
   auto result = Result();
   previous = result.resultWriter.shared.result->participate();

   auto r = make_unique<Q1>();
   auto lineitem = Scan("lineitem");
   Select(Expression().addOp(BF(primitives::sel_less_equal_Date_col_Date_val),
                             Buffer(sel_date, sizeof(pos_t)),
                             Column(lineitem, "l_shipdate"), Value(&r->c1)));
   Project()
       .addExpression(
           Expression()
               .addOp(conf.proj_sel_minus_int64_t_val_int64_t_col(),
                      Buffer(sel_date),
                      Buffer(result_proj_minus, sizeof(int64_t)),
                      Value(&r->one), Column(lineitem, "l_discount"))
               .addOp(conf.proj_multiplies_sel_int64_t_col_int64_t_col(),
                      Buffer(sel_date), Buffer(disc_price, sizeof(int64_t)),
                      Column(lineitem, "l_extendedprice"),
                      Buffer(result_proj_minus, sizeof(int64_t))))
       .addExpression(
           Expression()
               .addOp(conf.proj_sel_plus_int64_t_col_int64_t_val(),
                      Buffer(sel_date),
                      Buffer(result_proj_plus, sizeof(int64_t)),
                      Column(lineitem, "l_tax"), Value(&r->one))
               .addOp(conf.proj_multiplies_int64_t_col_int64_t_col(),
                      Buffer(charge, sizeof(int64_t)),
                      Buffer(disc_price, sizeof(int64_t)),
                      Buffer(result_proj_plus, sizeof(int64_t))));
   HashGroup()
       .pushKeySelVec(Buffer(sel_date), Buffer(sel_date_grouped, sizeof(pos_t)))
       .addKey(Column(lineitem, "l_returnflag"), Buffer(sel_date),
               primitives::hash_sel_Char_1_col,
               primitives::keys_not_equal_sel_Char_1_col,
               primitives::partition_by_key_sel_Char_1_col,
               Buffer(sel_date_grouped, sizeof(pos_t)),
               primitives::scatter_sel_Char_1_col,
               primitives::keys_not_equal_row_Char_1_col,
               primitives::partition_by_key_row_Char_1_col,
               primitives::scatter_sel_row_Char_1_col,
               primitives::gather_val_Char_1_col,
               Buffer(returnflag, sizeof(Char_1)))
       .addKey(Column(lineitem, "l_linestatus"), Buffer(sel_date),
               primitives::rehash_sel_Char_1_col,
               primitives::keys_not_equal_sel_Char_1_col,
               primitives::partition_by_key_sel_Char_1_col,
               Buffer(sel_date_grouped, sizeof(pos_t)),
               primitives::scatter_sel_Char_1_col,
               primitives::keys_not_equal_row_Char_1_col,
               primitives::partition_by_key_row_Char_1_col,
               primitives::scatter_sel_row_Char_1_col,
               primitives::gather_val_Char_1_col,
               Buffer(linestatus, sizeof(Char_1)))
       .padToAlign(sizeof(types::Numeric<12, 4>))
       .addValue(Buffer(disc_price), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_disc_price, sizeof(types::Numeric<12, 4>)))
       .addValue(Buffer(charge), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_charge, sizeof(types::Numeric<12, 4>)))
       .addValue(Column(lineitem, "l_quantity"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_qty, sizeof(types::Numeric<12, 2>)))
       .addValue(Column(lineitem, "l_extendedprice"), Buffer(sel_date),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_sel_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(sum_base_price, sizeof(types::Numeric<12, 2>)))
       .addValue(Buffer(charge, sizeof(uint64_t)),
                 primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_count_star,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(count_order, sizeof(uint64_t)));

   result.addValue("l_returnflag", Buffer(returnflag))
       .addValue("l_linestatus", Buffer(linestatus))
       .addValue("sum_qty", Buffer(sum_qty))
       .addValue("sum_base_price", Buffer(sum_base_price))
       .addValue("sum_disc_price", Buffer(sum_disc_price))
       .addValue("sum_charge", Buffer(sum_charge))
       .addValue("count_order", Buffer(count_order))
       .finalize();

   // TODO: add averages
   r->rootOp = popOperator();
   return r;
}

std::unique_ptr<runtime::Query> q1_vectorwise(Database& db, size_t nrThreads,
                                              size_t vectorSize) {
   using namespace vectorwise;
   WorkerGroup workers(nrThreads);
   vectorwise::SharedStateManager shared;

   std::unique_ptr<runtime::Query> result;
   workers.run([&]() {
      Q1Builder builder(db, shared, vectorSize);
      auto query = builder.getQuery();
      query->rootOp->next();
      auto leader = barrier();
      if (leader)
         result = move(
             dynamic_cast<ResultWriter*>(query->rootOp.get())->shared.result);
   });

   return result;
}
#endif
