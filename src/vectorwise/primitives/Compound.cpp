#ifdef VW_PROJ_COMPOUND
#include "vectorwise/CompoundPrimitives.hpp"
#include "vectorwise/Primitives.hpp"
#include <array>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace vectorwise {
namespace primitives {
namespace compound {

namespace {
struct Entry {
   const void* prim;
   Desc desc;
};
// every int64 projection primitive of the arithmetic ops compound covers
#define CP_FORMS(op, A)                                                        \
   {(const void*)proj_##op##_int64_t_col_int64_t_col, {A, Col, Col, false}},   \
       {(const void*)proj_##op##_int64_t_col_int64_t_val,                      \
        {A, Col, Val, false}},                                                 \
       {(const void*)proj_sel_both_##op##_int64_t_col_int64_t_col,             \
        {A, Sel, Sel, true}},                                                  \
       {(const void*)proj_##op##_sel_int64_t_col_int64_t_col,                  \
        {A, Sel, Col, true}},                                                  \
       {(const void*)proj_##op##_int64_t_col_sel_int64_t_col,                  \
        {A, Col, Sel, true}},                                                  \
       {(const void*)proj_sel_##op##_int64_t_col_int64_t_val,                  \
        {A, Sel, Val, true}},

// kernel table: index
// (((((op1 * 3 + a) * 3 + b) * 3 + op2) * 3 + c) * 2 + cLeft) * 2 + storeMid;
// only canonical chains are instantiated (432 of 972)
constexpr size_t kChains = 3 * 3 * 3 * 3 * 3 * 2 * 2;
constexpr Arith opAt(size_t i, size_t div) { return Arith(i / div % 3); }
constexpr Form formAt(size_t i, size_t div) { return Form(i / div % 3); }
constexpr bool canonical(size_t i) {
   const Arith op1 = opAt(i, 324), op2 = opAt(i, 12);
   const Form a = formAt(i, 108), b = formAt(i, 36);
   const bool cLeft = i / 2 % 2;
   return !(a == Val && b == Val) && !(op1 != Minus && a > b) &&
          !(op2 != Minus && cLeft);
}
template <size_t I> constexpr Fn entryFor() {
   if constexpr (canonical(I))
      return &compoundKernel<opAt(I, 324), formAt(I, 108), formAt(I, 36),
                             opAt(I, 12), formAt(I, 4), I / 2 % 2 == 1,
                             I % 2 == 1>;
   else
      return nullptr;
}
template <size_t... I>
constexpr std::array<Fn, sizeof...(I)> table(std::index_sequence<I...>) {
   return {entryFor<I>()...};
}
constexpr auto kernels = table(std::make_index_sequence<kChains>());
} // namespace

bool describe(const void* primitive, Desc& d) {
   // built on first use (plan building): the primitive pointers are globals
   // of Projection.cpp, initialized dynamically
   static const Entry entries[] = {
       CP_FORMS(plus, Plus) CP_FORMS(minus, Minus) CP_FORMS(multiplies, Mul)
       // constant first: non-commutative ops only
       {(const void*)proj_minus_int64_t_val_int64_t_col,
        {Minus, Val, Col, false}},
       {(const void*)proj_sel_minus_int64_t_val_int64_t_col,
        {Minus, Val, Sel, true}},
   };
   for (auto& e : entries)
      if (e.prim == primitive) {
         d = e.desc;
         return true;
      }
   return false;
}
#undef CP_FORMS

Fn kernel(Arith op1, Form a, Form b, Arith op2, Form c, bool cLeft,
          bool storeMid) {
   return kernels[(((((op1 * 3 + a) * 3 + b) * 3 + op2) * 3 + c) * 2 + cLeft) *
                      2 +
                  storeMid];
}

#ifdef VW_COMPOUND_CHAINS
std::string ChainSig::str() const {
   static const char* ops[] = {"plus", "minus", "mul"};
   static const char* forms[] = {"col", "sel", "val"};
   std::string r = std::string(ops[op0]) + " " + forms[a] + " " + forms[b];
   for (auto& s : steps)
      r += std::string(" ; ") + (s.storePrev ? "keep " : "reg ") + ops[s.op] +
           " " + forms[s.form] + (s.left ? " l" : " r");
   return r;
}

ChainFn findChain(const ChainSig& sig) {
   // built on first use (plan building)
   static const auto byName = [] {
      std::unordered_map<std::string_view, ChainFn> m;
      for (size_t i = 0; i < chainEntryCount; ++i)
         m.emplace(chainEntries[i].sig, chainEntries[i].fn);
      return m;
   }();
   auto it = byName.find(sig.str());
   return it == byName.end() ? nullptr : it->second;
}
#endif

} // namespace compound
} // namespace primitives
} // namespace vectorwise
#endif
