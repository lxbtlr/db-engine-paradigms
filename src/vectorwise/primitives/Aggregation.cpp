#include "common/runtime/Hash.hpp"
#include "common/runtime/SIMD.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <functional>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <x86intrin.h>
#else
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #include <simde/x86/avx512.h>
#endif

using namespace types;
using namespace std;

namespace vectorwise {
namespace primitives {

#define MK_AGGR_STATIC_COL(type, op)                                           \
   F2 aggr_static_##op##_##type##_col = (F2)&aggr_static_col<type, op>;

#define MK_AGGR_STATIC_SEL_COL(type, op)                                       \
   F3 aggr_static_sel_##op##_##type##_col = (F3)&aggr_static_sel_col<type, op>;
#define MK_AGGR_COL(type, op)                                                  \
   FAggr aggr_##op##_##type##_col = (FAggr)&aggr_col<type, op>;
#define MK_AGGR_SEL_COL(type, op)                                              \
   FAggrSel aggr_sel_##op##_##type##_col = (FAggrSel)&aggr_sel_col<type, op>;
#define MK_AGGR_ROW(type, op)                                                  \
   FAggrRow aggr_row_##op##_##type##_col = (FAggrRow)&aggr_row<type, op>;
#define MK_AGGR_INIT(type, op)                                                 \
   FAggrInit aggr_init_##op##_##type##_col = (FAggrInit)&aggr_init<type, op>;

pos_t aggr_static_count_star_(pos_t n, int64_t* RES result)
/// aggregate column
{
   *result += n;
   return n > 0;
}
F1 aggr_static_count_star = (F1)&aggr_static_count_star_;

pos_t aggr_count_star_(pos_t n, int64_t* RES entries[], void* RES /*param1*/,
                       size_t offset)
/// update count aggregates for each entry pointed to by entries
{
   for (uint64_t i = 0; i < n; ++i) {
      auto aggregate = addBytes(entries[i], offset);
      *aggregate += 1;
   }
   return n;
}
FAggr aggr_count_star = (FAggr)&aggr_count_star_;

EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_STATIC_COL)
EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_STATIC_SEL_COL)
EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_COL)
EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_SEL_COL)
EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_ROW)
EACH_ARITH(EACH_TYPE_FULL, MK_AGGR_INIT)

#if defined(__AVX512F__) && defined(__AVX512CD__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)

// AVX-512 aggregation with conflict detection for aggr_plus_int64_t_col.
// entries[i] is a pointer to the hash table entry; the actual aggregate
// lives at (char*)entries[i] + offset.  We gather 8 aggregate values,
// add 8 column values, detect pointer conflicts, and scatter back.
pos_t aggr_plus_int64_t_col_avx512_(pos_t n, int64_t* RES entries[],
                                     int64_t* RES param1, size_t offset) {
   uint64_t i = 0;
   size_t rest = n % 8;

   for (; i < n - rest; i += 8) {
      // Load 8 entry pointers
      auto ptrs = _mm512_loadu_si512((const void*)(entries + i));
      // Compute aggregate addresses: ptr + offset
      auto addrs = _mm512_add_epi64(ptrs, _mm512_set1_epi64(offset));

      // Gather current aggregate values from entry+offset
      auto aggregates = _mm512_i64gather_epi64(addrs, nullptr, 1);
      // Load 8 column values
      auto values = _mm512_loadu_si512((const void*)(param1 + i));
      // Add
      auto results = _mm512_add_epi64(aggregates, values);

      // Conflict detection on addresses
      auto conflicts = _mm512_conflict_epi64(addrs);
      // no_conflict_mask: lanes where conflict vector is zero (no prior lane has same address)
      auto no_conflict = _mm512_cmpeq_epi64_mask(conflicts, _mm512_setzero_si512());

      // Scatter non-conflicting lanes directly
      _mm512_mask_i64scatter_epi64(nullptr, no_conflict, addrs, results, 1);

      // Handle conflicting lanes scalar
      mask8_t conflict_mask = ~no_conflict & 0xFF;
      while (conflict_mask) {
         int lane = __builtin_ctz(conflict_mask);
         auto addr = reinterpret_cast<int64_t*>(
             reinterpret_cast<char*>(entries[i + lane]) + offset);
         *addr += param1[i + lane];
         conflict_mask &= conflict_mask - 1;
      }
   }
   // Scalar tail
   for (; i < (uint64_t)n; ++i) {
      auto aggregate = addBytes(entries[i], offset);
      *aggregate += param1[i];
   }
   return n;
}
FAggr aggr_plus_int64_t_col_avx512 = (FAggr)&aggr_plus_int64_t_col_avx512_;

// AVX-512 aggregation with conflict detection for aggr_sel_plus_int64_t_col.
// Same as above but param1 is indexed via a selection vector.
pos_t aggr_sel_plus_int64_t_col_avx512_(pos_t n, int64_t* RES entries[],
                                         pos_t* RES selParam1,
                                         int64_t* RES param1, size_t offset) {
   uint64_t i = 0;
   size_t rest = n % 8;

   for (; i < n - rest; i += 8) {
      // Load 8 entry pointers
      auto ptrs = _mm512_loadu_si512((const void*)(entries + i));
      // Compute aggregate addresses: ptr + offset
      auto addrs = _mm512_add_epi64(ptrs, _mm512_set1_epi64(offset));

      // Gather current aggregate values
      auto aggregates = _mm512_i64gather_epi64(addrs, nullptr, 1);

      // Gather column values via selection vector (sel indices are 32-bit)
      auto sel_indices = _mm256_loadu_si256((const __m256i*)(selParam1 + i));
      auto values = _mm512_i32gather_epi64(sel_indices, param1, 8);

      // Add
      auto results = _mm512_add_epi64(aggregates, values);

      // Conflict detection
      auto conflicts = _mm512_conflict_epi64(addrs);
      auto no_conflict = _mm512_cmpeq_epi64_mask(conflicts, _mm512_setzero_si512());

      // Scatter non-conflicting lanes
      _mm512_mask_i64scatter_epi64(nullptr, no_conflict, addrs, results, 1);

      // Handle conflicting lanes scalar
      mask8_t conflict_mask = ~no_conflict & 0xFF;
      while (conflict_mask) {
         int lane = __builtin_ctz(conflict_mask);
         auto aggregate = addBytes(entries[i + lane], offset);
         *aggregate += param1[selParam1[i + lane]];
         conflict_mask &= conflict_mask - 1;
      }
   }
   // Scalar tail
   for (; i < (uint64_t)n; ++i) {
      auto aggregate = addBytes(entries[i], offset);
      *aggregate += param1[selParam1[i]];
   }
   return n;
}
FAggrSel aggr_sel_plus_int64_t_col_avx512 =
    (FAggrSel)&aggr_sel_plus_int64_t_col_avx512_;

// AVX-512 count_star with conflict detection
pos_t aggr_count_star_avx512_(pos_t n, int64_t* RES entries[],
                               void* RES /*param1*/, size_t offset) {
   uint64_t i = 0;
   size_t rest = n % 8;
   auto ones = _mm512_set1_epi64(1);

   for (; i < n - rest; i += 8) {
      auto ptrs = _mm512_loadu_si512((const void*)(entries + i));
      auto addrs = _mm512_add_epi64(ptrs, _mm512_set1_epi64(offset));

      auto aggregates = _mm512_i64gather_epi64(addrs, nullptr, 1);
      auto results = _mm512_add_epi64(aggregates, ones);

      auto conflicts = _mm512_conflict_epi64(addrs);
      auto no_conflict = _mm512_cmpeq_epi64_mask(conflicts, _mm512_setzero_si512());

      _mm512_mask_i64scatter_epi64(nullptr, no_conflict, addrs, results, 1);

      mask8_t conflict_mask = ~no_conflict & 0xFF;
      while (conflict_mask) {
         int lane = __builtin_ctz(conflict_mask);
         auto aggregate = addBytes(entries[i + lane], offset);
         *aggregate += 1;
         conflict_mask &= conflict_mask - 1;
      }
   }
   for (; i < (uint64_t)n; ++i) {
      auto aggregate = addBytes(entries[i], offset);
      *aggregate += 1;
   }
   return n;
}
FAggr aggr_count_star_avx512 = (FAggr)&aggr_count_star_avx512_;

#endif
}
}
