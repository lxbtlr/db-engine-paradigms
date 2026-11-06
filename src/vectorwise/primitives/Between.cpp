// Range selection kernels and descriptors (BetweenPrimitives.hpp). Built in
// every configuration so tests and run_selbench reach the kernels; only the
// plan rewrite in ExpressionBuilder is gated by VW_SEL_BETWEEN.
#include "vectorwise/BetweenPrimitives.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdSelection.hpp"
#include <functional>

using namespace types;

namespace vectorwise {
namespace primitives {
namespace between {

namespace {

/// the kernel the selection primitives' options would pick: AVX-512 under
/// VW_SIMD_SEL (gathered input only with VW_SIMD_SEL_GATHER != scalar),
/// the branch-free template otherwise
template <typename T, template <typename> class Lo, template <typename> class Hi,
          bool SelIn>
pos_t run(pos_t n, pos_t* inSel, pos_t* out, void* col, void* lo, void* hi) {
   T* c = static_cast<T*>(col);
   T* l = static_cast<T*>(lo);
   T* h = static_cast<T*>(hi);
#ifdef VW_HAVE_SIMD_SEL
   if constexpr (!SelIn && simd::use_simd<T, Lo>() && simd::use_simd<T, Hi>())
      return simd::sel_between<T, Lo, Hi>(n, out, c, l, h);
   if constexpr (SelIn && simd::use_simd_gather<T, Lo>() &&
                 simd::use_simd_gather<T, Hi>())
      return simd::selsel_between<T, Lo, Hi>(n, inSel, out, c, l, h);
#endif
   return between_bf<T, Lo, Hi, SelIn>(n, inSel, out, c, l, h);
}

#define BT_SEL(T, Lo, Hi) &run<T, Lo, Hi, false>, &run<T, Lo, Hi, true>
#define BT_HI(T, Lo) BT_SEL(T, Lo, std::less_equal), BT_SEL(T, Lo, std::less)
#define BT_TYPE(T) BT_HI(T, std::greater_equal), BT_HI(T, std::greater)
// index ((type * 2 + loStrict) * 2 + hiStrict) * 2 + selInput
const Fn kernels[] = {BT_TYPE(int32_t), BT_TYPE(int64_t), BT_TYPE(Date)};
#undef BT_TYPE
#undef BT_HI
#undef BT_SEL

struct Entry {
   const void* prim;
   Desc desc;
};

} // namespace

#define BT_FORMS(T, TT, op, up, st)                                            \
   {(const void*)sel_##op##_##T##_col_##T##_val, {TT, up, st, false}},         \
       {(const void*)sel_##op##_##T##_col_##T##_val_bf, {TT, up, st, false}},  \
       {(const void*)selsel_##op##_##T##_col_##T##_val, {TT, up, st, true}},   \
       {(const void*)selsel_##op##_##T##_col_##T##_val_bf, {TT, up, st, true}},
#define BT_OPS(T, TT)                                                          \
   BT_FORMS(T, TT, greater_equal, false, false)                                \
   BT_FORMS(T, TT, greater, false, true)                                       \
   BT_FORMS(T, TT, less_equal, true, false)                                    \
   BT_FORMS(T, TT, less, true, true)

bool describe(const void* primitive, Desc& d) {
   // built on first use (plan building): the primitive pointers are globals
   // of Selection.cpp, initialized dynamically
   static const Entry entries[] = {
       BT_OPS(int32_t, Int32) BT_OPS(int64_t, Int64) BT_OPS(Date, Date32)};
   for (auto& e : entries)
      if (e.prim == primitive) {
         d = e.desc;
         return true;
      }
   return false;
}
#undef BT_OPS
#undef BT_FORMS

Fn kernel(Type t, bool loStrict, bool hiStrict, bool selInput) {
   return kernels[((t * 2 + loStrict) * 2 + hiStrict) * 2 + selInput];
}

} // namespace between
} // namespace primitives
} // namespace vectorwise
