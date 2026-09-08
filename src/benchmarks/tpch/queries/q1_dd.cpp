#include "benchmarks/tpch/DdKernel.hpp"
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
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>

using namespace runtime;
using namespace std;
using vectorwise::primitives::Char_1;
using vectorwise::primitives::hash_t;

// ---------------------------------------------------------------------------
// Data-dependency-width microbenchmark (Test 1) for Q1.
//
// The claim under test: the runtime gap between the compiled engine (Typer,
// q1_hyper) and the vectorized engine (Tectorwise, q1_vectorwise) on the
// cache-resident Q1 is driven by instruction count. The vectorized engine
// materializes every intermediate to memory, the compiled engine keeps them in
// registers. To strain this we inject W extra per-tuple intermediate values.
//
//   * Typer:   W is a compile-time constant (template<int W>) and the W
//              intermediates are distinct scalar SSA values in straight-line
//              code (recursive integer-sequence unrolling in DdKernel) -- real
//              register pressure, NOT an array with a runtime loop over W.
//   * Vectorized: W is a runtime count; each intermediate materializes into
//              its own buffer (that is correct and expected for this engine).
// ---------------------------------------------------------------------------

// Compile-time width set accepted by the Typer runtime dispatcher (each value
// below has an explicit instantiation in the switch in q1_dd_hyper).
[[maybe_unused]] constexpr int DD_WIDTHS[] = {2, 4, 6, 8,  10, 12, 14, 16,
                                             18, 20, 22, 24, 26, 28, 30, 32,
                                             48, 64};
constexpr int DD_WIDTHS_MAX = 64;

// Escape helper so a value must be considered live (prevents DCE of the
// synthetic work).
static void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

// ---------------------------------------------------------------------------
// Typer (hyper) variant -- compile-time W.
// Identical scan / l_shipdate filter / 4-group group-by / real aggregates as
// base q1_hyper; the synthetic kernel result is routed to a dedicated
// accumulator sink that is escaped but entirely separate from the real Q1
// aggregates, so Q1's reported answer is unchanged at every W.
// ---------------------------------------------------------------------------
template <int W, dd::Shape S>
NOVECTORIZE std::unique_ptr<runtime::Query> q1_dd_hyper_impl(
    runtime::Database& db, size_t nrThreads) {
   using Kernel = typename dd::Kernel<W, S>::type;
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

   // Dedicated live-out sink for the synthetic work. Separate from the real Q1
   // aggregates; escaped below so the kernel cannot be dead-code eliminated.
   std::atomic<int64_t> sink{0};

   tbb::parallel_for(
       tbb::blocked_range<size_t>(0, li.nrTuples, morselSize),
       [&](const tbb::blocked_range<size_t>& r) {
          auto locals = groupOp.preAggLocals();
          int64_t localSink = 0;
          for (size_t i = r.begin(), end = r.end(); i != end; ++i) {
             if (l_shipdate[i] <= c1) {
                auto& group = locals.getGroup(make_tuple(l_returnflag[i],
                                                         l_linestatus[i]));

                get<0>(group) += l_quantity[i];
                get<1>(group) += l_extendedprice[i];
                auto disc_price = l_extendedprice[i] * (one - l_discount[i]);
                get<2>(group) += disc_price;
                auto charge = disc_price * (one + l_tax[i]);
                get<3>(group) += charge;
                get<4>(group) += 1;

                // Synthetic data-dependency work: W distinct scalar SSA locals
                // reduced to a single value routed to the sink. The kernel
                // shape (chained vs independent) is a compile-time template
                // parameter.
                localSink += Kernel::run(l_extendedprice[i].value,
                                         l_discount[i].value);
             }
          }
          sink.fetch_add(localSink, std::memory_order_relaxed);
       });

   escape(&sink);  // force the synthetic work to be live-out

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

   groupOp.forallGroups(
       [&](runtime::Stack<typename decltype(groupOp)::group_t>& /*auto&*/ entries) {
          auto n = entries.size();
          auto block = result->createBlock(n);
          auto ret = reinterpret_cast<Char<1>*>(block.data(retAttr));
          auto status = reinterpret_cast<Char<1>*>(block.data(statusAttr));
          auto qty = reinterpret_cast<Numeric<12, 2>*>(block.data(qtyAttr));
          auto base_price =
              reinterpret_cast<Numeric<12, 2>*>(block.data(base_priceAttr));
          auto disc_price =
              reinterpret_cast<Numeric<12, 4>*>(block.data(disc_priceAttr));
          auto charge =
              reinterpret_cast<Numeric<12, 6>*>(block.data(chargeAttr));
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

template <dd::Shape S>
std::unique_ptr<runtime::Query> dispatchW(runtime::Database& db,
                                          size_t nrThreads, int W) {
   switch (W) {
      case 2: return q1_dd_hyper_impl<2, S>(db, nrThreads);
      case 4: return q1_dd_hyper_impl<4, S>(db, nrThreads);
      case 6: return q1_dd_hyper_impl<6, S>(db, nrThreads);
      case 8: return q1_dd_hyper_impl<8, S>(db, nrThreads);
      case 10: return q1_dd_hyper_impl<10, S>(db, nrThreads);
      case 12: return q1_dd_hyper_impl<12, S>(db, nrThreads);
      case 14: return q1_dd_hyper_impl<14, S>(db, nrThreads);
      case 16: return q1_dd_hyper_impl<16, S>(db, nrThreads);
      case 18: return q1_dd_hyper_impl<18, S>(db, nrThreads);
      case 20: return q1_dd_hyper_impl<20, S>(db, nrThreads);
      case 22: return q1_dd_hyper_impl<22, S>(db, nrThreads);
      case 24: return q1_dd_hyper_impl<24, S>(db, nrThreads);
      case 26: return q1_dd_hyper_impl<26, S>(db, nrThreads);
      case 28: return q1_dd_hyper_impl<28, S>(db, nrThreads);
      case 30: return q1_dd_hyper_impl<30, S>(db, nrThreads);
      case 32: return q1_dd_hyper_impl<32, S>(db, nrThreads);
      case 48: return q1_dd_hyper_impl<48, S>(db, nrThreads);
      case 64: return q1_dd_hyper_impl<64, S>(db, nrThreads);
      default:
         std::cerr << "q1_dd_hyper: unsupported W=" << W
                   << " (must be one of DD_WIDTHS)\n";
         exit(1);
   }
}

std::unique_ptr<runtime::Query> q1_dd_hyper(runtime::Database& db,
                                            size_t nrThreads, int W,
                                            dd::Shape shape) {
   switch (shape) {
      case dd::Shape::Chained:
         return dispatchW<dd::Shape::Chained>(db, nrThreads, W);
      case dd::Shape::Independent:
         return dispatchW<dd::Shape::Independent>(db, nrThreads, W);
      default:
         std::cerr << "q1_dd_hyper: unsupported kernel shape\n";
         exit(1);
   }
}

// ---------------------------------------------------------------------------
// Vectorized (Tectorwise) variant -- runtime W.
// Identical pipeline to base q1_vectorwise (getQuery), but the projection chain
// is extended with W intermediate expressions (each materializing intermediate
// k into its own buffer from a column and the previous buffer) plus a final
// expression that sums all W into the dedicated escaped sink. The real Q1
// aggregates and result columns are unchanged; the sink is emitted as an extra
// result column so the harness escape keeps it live.
// ---------------------------------------------------------------------------
std::unique_ptr<Q1Builder::Q1> Q1Builder::getQueryDd(int W, dd::Shape shape) {
   using namespace vectorwise;
   if (W < 1 || W > DD_WIDTHS_MAX) {
      std::cerr << "q1_dd_vectorwise: unsupported W=" << W
                << " (must be in [1," << DD_WIDTHS_MAX << "])\n";
      exit(1);
   }
   if (shape != dd::Shape::Chained && shape != dd::Shape::Independent) {
      std::cerr << "q1_dd_vectorwise: unsupported kernel shape\n";
      exit(1);
   }

   auto result = Result();
   previous = result.resultWriter.shared.result->participate();

   auto r = make_unique<Q1>();
   for (int k = 0; k < W; ++k) r->ddC[k] = dd::ddConstant(k);

   auto lineitem = Scan("lineitem");
   Select(Expression().addOp(BF(primitives::sel_less_equal_Date_col_Date_val),
                             Buffer(sel_date, sizeof(pos_t)),
                             Column(lineitem, "l_shipdate"), Value(&r->c1)));

   auto proj = Project();
   proj.addExpression(
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

   // --- W synthetic intermediates, each into its own buffer ---
   // Shape is a runtime count here (each intermediate materializes to memory,
   // so on this engine the two shapes are ~cost-neutral by design).
   if (shape == dd::Shape::Independent) {
      // Independent DAG: every intermediate is computed directly from the
      // columns (extendedprice * Ck + extendedprice), none reads a previous
      // intermediate. Same per-step op mix (multiply + add) as chained.
      for (int k = 0; k < W; ++k) {
         proj.addExpression(
             Expression()
                 .addOp(primitives::proj_sel_multiplies_int64_t_col_int64_t_val,
                        Buffer(sel_date), Buffer(dd_scratch, sizeof(int64_t)),
                        Column(lineitem, "l_extendedprice"), Value(&r->ddC[k]))
                 .addOp(primitives::proj_plus_sel_int64_t_col_int64_t_col,
                        Buffer(sel_date),
                        Buffer(dd_buf_base + k, sizeof(int64_t)),
                        Buffer(dd_scratch, sizeof(int64_t)),
                        Column(lineitem, "l_extendedprice")));
      }
   } else {
      // Chained DAG (unchanged from Test 1): buf[0] = ep*C0;
      // buf[k] = buf[k-1] * Ck + extendedprice (reads the previous buffer).
      // buf[0] = extendedprice * C0                       (col * val)
      proj.addExpression(
          Expression()
              .addOp(primitives::proj_sel_multiplies_int64_t_col_int64_t_val,
                     Buffer(sel_date), Buffer(dd_buf_base + 0, sizeof(int64_t)),
                     Column(lineitem, "l_extendedprice"), Value(&r->ddC[0])));
      for (int k = 1; k < W; ++k) {
         proj.addExpression(
             Expression()
                 .addOp(primitives::proj_sel_multiplies_int64_t_col_int64_t_val,
                        Buffer(sel_date), Buffer(dd_scratch, sizeof(int64_t)),
                        Buffer(dd_buf_base + k - 1, sizeof(int64_t)),
                        Value(&r->ddC[k]))
                 .addOp(primitives::proj_plus_sel_int64_t_col_int64_t_col,
                        Buffer(sel_date),
                        Buffer(dd_buf_base + k, sizeof(int64_t)),
                        Buffer(dd_scratch, sizeof(int64_t)),
                        Column(lineitem, "l_extendedprice")));
      }
   }
   // --- final: sink = sum of all W intermediates ---
   proj.addExpression(
       Expression()
           .addOp(primitives::proj_sel_plus_int64_t_col_int64_t_val,
                  Buffer(sel_date), Buffer(dd_sink, sizeof(int64_t)),
                  Buffer(dd_buf_base + 0, sizeof(int64_t)), Value(&r->ddZero)));
   for (int k = 1; k < W; ++k) {
      proj.addExpression(
          Expression()
              .addOp(primitives::proj_plus_sel_int64_t_col_int64_t_col,
                     Buffer(sel_date), Buffer(dd_sink, sizeof(int64_t)),
                     Buffer(dd_sink, sizeof(int64_t)),
                     Buffer(dd_buf_base + k, sizeof(int64_t))));
   }

   OptHashGroup()
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
                 Buffer(count_order, sizeof(uint64_t)))
       // The dedicated synthetic sink, aggregated into its own out buffer.
       // Separate from all real Q1 aggregates; emitted below as an extra column
       // so the harness escape keeps it live.
       .addValue(Buffer(dd_sink), primitives::aggr_init_plus_int64_t_col,
                 primitives::aggr_plus_int64_t_col,
                 primitives::aggr_row_plus_int64_t_col,
                 primitives::gather_val_int64_t_col,
                 Buffer(dd_sink_out, sizeof(int64_t)));

   result.addValue("l_returnflag", Buffer(returnflag))
       .addValue("l_linestatus", Buffer(linestatus))
       .addValue("sum_qty", Buffer(sum_qty))
       .addValue("sum_base_price", Buffer(sum_base_price))
       .addValue("sum_disc_price", Buffer(sum_disc_price))
       .addValue("sum_charge", Buffer(sum_charge))
       .addValue("count_order", Buffer(count_order))
       .addValue("dd_sink", Buffer(dd_sink_out))
       .finalize();

   r->rootOp = popOperator();
   return r;
}

std::unique_ptr<runtime::Query> q1_dd_vectorwise(runtime::Database& db,
                                                 size_t nrThreads,
                                                 size_t vectorSize, int W,
                                                 dd::Shape shape) {
   using namespace vectorwise;
   WorkerGroup workers(nrThreads);
   vectorwise::SharedStateManager shared;

   std::unique_ptr<runtime::Query> result;
   workers.run([&]() {
      Q1Builder builder(db, shared, vectorSize);
      auto query = builder.getQueryDd(W, shape);
      query->rootOp->next();
      auto leader = barrier();
      if (leader)
         result = move(
             dynamic_cast<ResultWriter*>(query->rootOp.get())->shared.result);
   });

   return result;
}
