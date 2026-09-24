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
#if defined(HYPER_Q1_SIMD) && defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#include <immintrin.h>
#define HYPER_Q1_SIMD_ACTIVE
#endif

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

static void dumpQ1Result(const char* label, runtime::Query* query) {
   if (!query || !query->result) return;
   auto& rel = *query->result;
   auto retAttr = rel.getAttribute("l_returnflag");
   auto statusAttr = rel.getAttribute("l_linestatus");
   auto qtyAttr = rel.getAttribute("sum_qty");
   auto basePriceAttr = rel.getAttribute("sum_base_price");
   auto discPriceAttr = rel.getAttribute("sum_disc_price");
   auto chargeAttr = rel.getAttribute("sum_charge");
   auto countAttr = rel.getAttribute("count_order");

   fprintf(stderr, "\n=== Q1 Results [%s] ===\n", label);
   fprintf(stderr, "%-4s %-4s %20s %20s %20s %20s %15s\n",
           "ret", "stat", "sum_qty", "sum_base_price", "sum_disc_price",
           "sum_charge", "count_order");
   for (auto& block : rel) {
      auto n = block.size();
      auto ret = reinterpret_cast<types::Char<1>*>(block.data(retAttr));
      auto status = reinterpret_cast<types::Char<1>*>(block.data(statusAttr));
      auto qty = reinterpret_cast<int64_t*>(block.data(qtyAttr));
      auto basePrice = reinterpret_cast<int64_t*>(block.data(basePriceAttr));
      auto discPrice = reinterpret_cast<int64_t*>(block.data(discPriceAttr));
      auto charge = reinterpret_cast<int64_t*>(block.data(chargeAttr));
      auto count = reinterpret_cast<int64_t*>(block.data(countAttr));
      for (size_t i = 0; i < n; ++i) {
         fprintf(stderr, "%-4c %-4c %20ld %20ld %20ld %20ld %15ld\n",
                 ret[i].value, status[i].value,
                 qty[i], basePrice[i], discPrice[i], charge[i], count[i]);
      }
   }
   fprintf(stderr, "=== END ===\n\n");
}



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

#ifdef HYPER_Q1_DIRECT_AGG
   // Direct-mapped aggregation: bypass hash table for the per-morsel hot loop.
   // l_returnflag in {A,N,R}, l_linestatus in {F,O} => 6 groups, 8 slots.
   // Slot = flagIndex*2 + statusBit. One hash lookup per slot at morsel end.
   static constexpr int NUM_SLOTS = 8;

   // Map returnflag char to slot index: A->0, N->1, R->2
   auto flagIndex = [](unsigned char c) -> unsigned {
      return (c == 'A') ? 0u : (c == 'N') ? 1u : 2u;
   };

   // Raw int32_t pointers for SIMD (Date is int32_t wrapper)
   auto shipdate_raw = reinterpret_cast<const int32_t*>(l_shipdate);
   // Numeric<12,2> / Char<1> data pointers cast for direct access
   auto ep_raw = reinterpret_cast<const int64_t*>(l_extendedprice);
   auto disc_raw = reinterpret_cast<const int64_t*>(l_discount);
   auto tax_raw = reinterpret_cast<const int64_t*>(l_tax);
   auto qty_raw = reinterpret_cast<const int64_t*>(l_quantity);
   // Char<1> is {uint8_t len, char value} = 2 bytes per element
   auto flag_raw = reinterpret_cast<const uint8_t*>(l_returnflag);
   auto status_raw = reinterpret_cast<const uint8_t*>(l_linestatus);
   int32_t c1_raw = c1.value;

   tbb::parallel_for(
       tbb::blocked_range<size_t>(0, li.nrTuples, morselSize),
       [&](const tbb::blocked_range<size_t>& r) {
          auto locals = groupOp.preAggLocals();

          int64_t qty_acc[NUM_SLOTS] = {};
          int64_t base_price_acc[NUM_SLOTS] = {};
          int64_t disc_price_acc[NUM_SLOTS] = {};
          int64_t charge_acc[NUM_SLOTS] = {};
          int64_t count[NUM_SLOTS] = {};

          size_t begin = r.begin(), end = r.end();

#ifdef HYPER_Q1_SIMD_ACTIVE
          // AVX-512 fused scan + aggregate: 8 int64_t lanes per iteration.
          // Date comparison uses 256-bit (8 x int32), data uses 512-bit (8 x int64).
          // For each vector of 8 tuples, build per-slot masks from flag/status
          // comparisons and do masked adds into per-slot ZMM accumulators.

          // Per-slot vector accumulators (6 used slots out of 8)
          // Layout: [A-F, A-O, N-F, N-O, R-F, R-O, unused, unused]
          __m512i v_qty[NUM_SLOTS], v_bp[NUM_SLOTS], v_dp[NUM_SLOTS],
                  v_ch[NUM_SLOTS], v_cnt[NUM_SLOTS];
          for (int s = 0; s < NUM_SLOTS; ++s) {
             v_qty[s] = _mm512_setzero_si512();
             v_bp[s]  = _mm512_setzero_si512();
             v_dp[s]  = _mm512_setzero_si512();
             v_ch[s]  = _mm512_setzero_si512();
             v_cnt[s] = _mm512_setzero_si512();
          }

          __m256i c1_vec = _mm256_set1_epi32(c1_raw);
          __m512i v100 = _mm512_set1_epi64(100);
          __m512i v_one = _mm512_set1_epi64(1);

          // Flag/status broadcast values for comparison
          // Char<1> = {len, value}, 2 bytes. To extract 8 value bytes:
          // load 16 bytes, shuffle odd positions to low 8, zero-extend to epi32
          __m128i shuf_mask = _mm_set_epi8(
              -1,-1,-1,-1,-1,-1,-1,-1, 15,13,11,9,7,5,3,1);
          __m256i vA = _mm256_set1_epi32('A');
          __m256i vN = _mm256_set1_epi32('N');
          __m256i vR = _mm256_set1_epi32('R');
          __m256i vF = _mm256_set1_epi32('F');
          __m256i vO = _mm256_set1_epi32('O');

          size_t i = begin;
          size_t simd_end = begin + ((end - begin) / 8) * 8;

          for (; i < simd_end; i += 8) {
             // Date filter: load 8 x int32, compare <= c1
             __m256i dates = _mm256_loadu_si256(
                 (const __m256i*)(shipdate_raw + i));
             __mmask8 dm = _mm256_cmple_epi32_mask(dates, c1_vec);
             if (dm == 0) continue;

             // Load 8 x int64 columns
             __m512i ep  = _mm512_loadu_si512(ep_raw + i);
             __m512i dsc = _mm512_loadu_si512(disc_raw + i);
             __m512i tx  = _mm512_loadu_si512(tax_raw + i);
             __m512i qt  = _mm512_loadu_si512(qty_raw + i);

             // disc_price = ep * (100 - discount)
             __m512i dp = _mm512_mullo_epi64(
                 ep, _mm512_sub_epi64(v100, dsc));
             // charge = disc_price * (100 + tax)
             __m512i ch = _mm512_mullo_epi64(
                 dp, _mm512_add_epi64(v100, tx));

             // Extract flag value bytes: load 16 bytes of Char<1>[8],
             // shuffle to get value bytes, widen to epi32 for comparison
             __m128i flags_raw = _mm_loadu_si128(
                 (const __m128i*)(flag_raw + i * 2));
             __m128i flag_bytes = _mm_shuffle_epi8(flags_raw, shuf_mask);
             __m256i flags = _mm256_cvtepu8_epi32(flag_bytes);

             __m128i stats_raw = _mm_loadu_si128(
                 (const __m128i*)(status_raw + i * 2));
             __m128i stat_bytes = _mm_shuffle_epi8(stats_raw, shuf_mask);
             __m256i stats = _mm256_cvtepu8_epi32(stat_bytes);

             // Build per-slot masks: slot = flagIdx*2 + statusIdx
             __mmask8 mA = _mm256_cmpeq_epi32_mask(flags, vA);
             __mmask8 mN = _mm256_cmpeq_epi32_mask(flags, vN);
             __mmask8 mR = _mm256_cmpeq_epi32_mask(flags, vR);
             __mmask8 mF = _mm256_cmpeq_epi32_mask(stats, vF);
             __mmask8 mO = _mm256_cmpeq_epi32_mask(stats, vO);

             // 6 slot masks, each AND'd with date mask
             __mmask8 sm[6];
             sm[0] = dm & mA & mF;  // A-F
             sm[1] = dm & mA & mO;  // A-O
             sm[2] = dm & mN & mF;  // N-F
             sm[3] = dm & mN & mO;  // N-O
             sm[4] = dm & mR & mF;  // R-F
             sm[5] = dm & mR & mO;  // R-O

             // Masked accumulate into per-slot vector registers
             for (int s = 0; s < 6; ++s) {
                if (sm[s] == 0) continue;
                v_qty[s] = _mm512_mask_add_epi64(v_qty[s], sm[s], v_qty[s], qt);
                v_bp[s]  = _mm512_mask_add_epi64(v_bp[s],  sm[s], v_bp[s],  ep);
                v_dp[s]  = _mm512_mask_add_epi64(v_dp[s],  sm[s], v_dp[s],  dp);
                v_ch[s]  = _mm512_mask_add_epi64(v_ch[s],  sm[s], v_ch[s],  ch);
                v_cnt[s] = _mm512_mask_add_epi64(v_cnt[s], sm[s], v_cnt[s], v_one);
             }
          }

          // Horizontal reduce vector accumulators into scalar arrays
          for (int s = 0; s < 6; ++s) {
             qty_acc[s]        = _mm512_reduce_add_epi64(v_qty[s]);
             base_price_acc[s] = _mm512_reduce_add_epi64(v_bp[s]);
             disc_price_acc[s] = _mm512_reduce_add_epi64(v_dp[s]);
             charge_acc[s]     = _mm512_reduce_add_epi64(v_ch[s]);
             count[s]          = _mm512_reduce_add_epi64(v_cnt[s]);
          }

          // Scalar tail
          for (; i < end; ++i) {
             if (shipdate_raw[i] <= c1_raw) {
                unsigned char flag = flag_raw[i * 2 + 1];
                unsigned char status = status_raw[i * 2 + 1];
                unsigned slot = flagIndex(flag) * 2 + ((status >> 3) & 1);
                int64_t ep = ep_raw[i];
                int64_t disc = disc_raw[i];
                int64_t tx = tax_raw[i];
                int64_t dp_val = ep * (100 - disc);
                int64_t ch_val = dp_val * (100 + tx);
                qty_acc[slot] += qty_raw[i];
                base_price_acc[slot] += ep;
                disc_price_acc[slot] += dp_val;
                charge_acc[slot] += ch_val;
                count[slot] += 1;
             }
          }
#else
          // Scalar direct-mapped loop
          for (size_t i = begin; i < end; ++i) {
             if (l_shipdate[i].value <= c1.value) {
                unsigned char flag = l_returnflag[i].value;
                unsigned char status = l_linestatus[i].value;
                unsigned slot = flagIndex(flag) * 2 + ((status >> 3) & 1);

                assert(flag == 'A' || flag == 'N' || flag == 'R');
                assert(status == 'F' || status == 'O');

                int64_t ep = l_extendedprice[i].value;
                int64_t disc = l_discount[i].value;
                int64_t tx = l_tax[i].value;
                int64_t dp = ep * (100 - disc);
                int64_t ch = dp * (100 + tx);

                qty_acc[slot] += l_quantity[i].value;
                base_price_acc[slot] += ep;
                disc_price_acc[slot] += dp;
                charge_acc[slot] += ch;
                count[slot] += 1;
             }
          }
#endif // HYPER_Q1_SIMD_ACTIVE

          // Flush non-empty slots into the hash table (at most 6 calls)
          static constexpr char flagChars[3] = {'A', 'N', 'R'};
          static constexpr char statusChars[2] = {'F', 'O'};
          for (int fi = 0; fi < 3; ++fi) {
             for (int si = 0; si < 2; ++si) {
                int slot = fi * 2 + si;
                if (count[slot] == 0) continue;
                Char<1> rf; rf.len = 1; rf.value[0] = flagChars[fi];
                Char<1> ls; ls.len = 1; ls.value[0] = statusChars[si];
                auto& group = locals.getGroup(make_tuple(rf, ls));
                get<0>(group).value += qty_acc[slot];
                get<1>(group).value += base_price_acc[slot];
                get<2>(group).value += disc_price_acc[slot];
                get<3>(group).value += charge_acc[slot];
                get<4>(group) += count[slot];
             }
          }
       });
#else
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
#endif

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

   groupOp.forallGroups([&](runtime::Stack<decltype(groupOp)::group_t>& /*auto&*/ entries) {
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
   Select(Expression().addOp(conf.sel_less_equal_int32_t_col_int32_t_val(),
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
      /* auto found = */ query->rootOp->next();
      auto leader = barrier();
      if (leader)
         result = move(
             dynamic_cast<ResultWriter*>(query->rootOp.get())->shared.result);
   });
   //dumpQ1Result("vectorwise", result.get());
   return result;
}
