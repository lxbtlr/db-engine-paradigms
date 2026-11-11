#pragma once
// Range selection primitives (VW_SEL_COMPOUND, X100-style compound selection).
// Inside one Select expression, two comparisons of the same column with
// constants, one a lower bound (>=, >) and one an upper bound (<=, <), where
// the second reads the first's output selection, run as one pass:
//    out = positions idx (of inSel, or 0..n-1) with lo(x[idx], l) && hi(x[idx], h)
// The column is read once and the selection vector between the two
// comparisons is not written. ExpressionBuilder finds the pairs at build time
// (QueryBuilder.cpp); kernel() picks the scalar template below or the AVX-512
// kernels in SimdSelection.hpp under the same options as the selection
// primitives the pair replaces (Between.cpp).
#include <cstdint>
#include "vectorwise/defs.hpp"

namespace vectorwise {
namespace primitives {
namespace between {

enum Type : uint8_t { Int32, Int64, Date32 };

/// what a sel_/selsel_ comparison of a column with a constant computes
struct Desc {
   Type type;
   bool upper;    // <=, < (else >=, >)
   bool strict;   // <, > (else <=, >=)
   bool selInput; // F4: (n, inSel, result, col, val); else F3: (n, result, col, val)
};
/// Desc of a selection primitive; false if it is not an int32, int64 or Date
/// ordering comparison of a column with a constant (equality, col_col, Char,
/// the *_avx512 kernels, ...)
bool describe(const void* primitive, Desc& d);

/// out may be inSel: every kernel writes at or behind the entry it reads
using Fn = pos_t (*)(pos_t n, pos_t* inSel, pos_t* out, void* col, void* lo,
                     void* hi);
Fn kernel(Type t, bool loStrict, bool hiStrict, bool selInput);

/// scalar branch-free kernel (no RES on inSel/out: they may alias)
template <typename T, template <typename> class Lo, template <typename> class Hi,
          bool SelIn>
pos_t between_bf(pos_t n, pos_t* inSel, pos_t* out, T* RES col, T* RES lo,
                 T* RES hi) {
   const T l = *lo, h = *hi;
   pos_t found = 0;
   for (pos_t i = 0; i < n; i++) {
      const pos_t idx = SelIn ? inSel[i] : i;
      const T v = col[idx];
      out[found] = idx;
      found += Lo<T>()(v, l) & Hi<T>()(v, h);
   }
   return found;
}

} // namespace between
} // namespace primitives
} // namespace vectorwise
