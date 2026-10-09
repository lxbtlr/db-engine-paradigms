#pragma once
// VW_PROJ_COMPOUND: compound projection primitives (X100-style). Inside one
// Project expression, an int64 +, -, * projection whose result feeds the next
// projection is evaluated with it in one loop:
//    mid[i] = op1(a, b);   out[i] = op2(mid[i], c)  (or op2(c, mid[i]))
// where a, b and c are each a dense column (Col), a column read through the
// expression's selection vector (Sel) or a constant (Val). The selection is
// loaded once per row; mid is still written, so the buffers hold what the
// two separate primitives would have written. ExpressionBuilder finds the
// chains at build time (QueryBuilder.cpp); one template body, instantiated
// for the canonical forms in Compound.cpp.
#include <cstdint>
#include <functional>
#include "vectorwise/defs.hpp"

namespace vectorwise {
namespace primitives {
namespace compound {

enum Form : uint8_t { Col, Sel, Val };
enum Arith : uint8_t { Plus, Minus, Mul };
inline bool commutative(Arith a) { return a != Minus; }

/// what an int64 +, -, * projection primitive computes
struct Desc {
   Arith op;
   Form a, b;     // result[i] = op(a, b)
   bool selInput; // F4: (n, sel, result, p1, p2); else F3: (n, result, p1, p2)
};
/// Desc of a projection primitive; false if it is not an int64 +, -, *
/// projection (other types, other ops, the SIMD kernels, selections, ...)
bool describe(const void* primitive, Desc& d);

using Fn = pos_t (*)(pos_t n, const pos_t* sel, int64_t* mid, int64_t* out,
                     const int64_t* a, const int64_t* b, const int64_t* c);
/// the kernel for a canonical chain: a <= b (in Form order) if op1 is
/// commutative, !cLeft if op2 is commutative, not both a and b Val;
/// nullptr otherwise
Fn kernel(Arith op1, Form a, Form b, Arith op2, Form c, bool cLeft);

template <Arith> struct ArithOp;
template <> struct ArithOp<Plus> { using T = std::plus<int64_t>; };
template <> struct ArithOp<Minus> { using T = std::minus<int64_t>; };
template <> struct ArithOp<Mul> { using T = std::multiplies<int64_t>; };

template <Form F>
inline int64_t at(const int64_t* RES p, int64_t v, pos_t i, pos_t s) {
   if constexpr (F == Col) return p[i];
   else if constexpr (F == Sel) return p[s];
   else return v;
}

template <Arith Op1, Form A, Form B, Arith Op2, Form C, bool CLeft>
pos_t compoundKernel(pos_t n, const pos_t* RES sel, int64_t* RES mid,
                     int64_t* RES out, const int64_t* RES a,
                     const int64_t* RES b, const int64_t* RES c) {
   constexpr bool anySel = A == Sel || B == Sel || C == Sel;
   const int64_t va = A == Val ? *a : 0, vb = B == Val ? *b : 0,
                 vc = C == Val ? *c : 0;
   typename ArithOp<Op1>::T op1;
   typename ArithOp<Op2>::T op2;
   for (pos_t i = 0; i < n; i++) {
      pos_t s = i;
      if constexpr (anySel) s = sel[i];
      const int64_t m = op1(at<A>(a, va, i, s), at<B>(b, vb, i, s));
      const int64_t x = at<C>(c, vc, i, s);
      mid[i] = m;
      out[i] = CLeft ? op2(x, m) : op2(m, x);
   }
   return n;
}

} // namespace compound
} // namespace primitives
} // namespace vectorwise
