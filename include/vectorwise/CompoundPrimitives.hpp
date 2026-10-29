#pragma once
// VW_PROJ_COMPOUND: compound projection primitives (X100-style). Inside one
// Project expression, an int64 +, -, * projection whose result feeds the next
// projection is evaluated with it in one loop:
//    mid[i] = op1(a, b);   out[i] = op2(mid[i], c)  (or op2(c, mid[i]))
// where a, b and c are each a dense column (Col), a column read through the
// expression's selection vector (Sel) or a constant (Val). The selection is
// loaded once per row. mid is written, so the buffers hold what the two
// separate primitives would have written, unless it is a Scratch buffer
// (QueryBuilder::Scratch: read by nothing but the second op): then the
// kernel keeps it in a register and stores only out (StoreMid = false).
// ExpressionBuilder finds the chains at build time (QueryBuilder.cpp); one
// template body, instantiated for the canonical forms in Compound.cpp.
//
// Longer chains (VW_COMPOUND_DEPTH > 2, or VW_COMPOUND_SET=workload): up to
// VW_COMPOUND_DEPTH projections, each reading the previous one's result,
// run as one chainKernel. Only the chains listed for the build are compiled
// (cmake/CompoundKernels.cmake: all canonical chains up to depth 3, or the
// recorded workload in compound_workload.txt); the builder splits a chain
// without a kernel into the longest pieces that have one.
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
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
Fn kernel(Arith op1, Form a, Form b, Arith op2, Form c, bool cLeft,
          bool storeMid);

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

template <Arith Op1, Form A, Form B, Arith Op2, Form C, bool CLeft,
          bool StoreMid>
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
      if constexpr (StoreMid) mid[i] = m;
      out[i] = CLeft ? op2(x, m) : op2(m, x);
   }
   return n;
}

#ifdef VW_COMPOUND_CHAINS
/// step k of a chain (k >= 1): acc = op(acc, x), or op(x, acc) if Left, with
/// x the step's operand; StorePrev: first store acc (intermediate k) to its
/// buffer. Canonical: !Left if Op is commutative.
template <Arith Op, Form F, bool Left, bool StorePrev> struct Step {
   static constexpr Arith op = Op;
   static constexpr Form form = F;
   static constexpr bool left = Left, storePrev = StorePrev;
};
struct StepSig {
   Arith op;
   Form form;
   bool left, storePrev;
};
/// a chain: op0(a, b) followed by steps; str() is the name used by the kernel
/// lists, e.g. "mul col val ; reg plus sel r" (reg: the intermediate stays in
/// a register, keep: it is stored)
struct ChainSig {
   Arith op0;
   Form a, b;
   std::vector<StepSig> steps;
   std::string str() const;
};
/// in: the operands, a and b then one per step (a Val operand points to its
/// value); mid: per step, the buffer its incoming intermediate is stored to
/// (unused if !storePrev)
using ChainFn = pos_t (*)(pos_t n, const pos_t* sel, int64_t* const* mid,
                          int64_t* out, const int64_t* const* in);
/// the compiled kernel for a canonical chain of this build, or nullptr
ChainFn findChain(const ChainSig& sig);
/// the build's kernel list, generated by cmake/CompoundKernels.cmake
struct ChainEntry {
   const char* sig;
   ChainFn fn;
};
extern const ChainEntry chainEntries[];
extern const size_t chainEntryCount;

template <typename St>
inline int64_t chainStep(int64_t acc, const int64_t* RES p, int64_t v,
                         int64_t* RES m, pos_t i, pos_t s) {
   if constexpr (St::storePrev) m[i] = acc;
   typename ArithOp<St::op>::T op;
   const int64_t x = at<St::form>(p, v, i, s);
   return St::left ? op(x, acc) : op(acc, x);
}

template <Arith Op0, Form A, Form B, typename... Steps> struct Chain {
   static constexpr bool anySel =
       A == Sel || B == Sel || ((Steps::form == Sel) || ...);
   template <size_t... I>
   static pos_t run(pos_t n, const pos_t* RES sel, int64_t* const* mid,
                    int64_t* RES out, const int64_t* const* in,
                    std::index_sequence<I...>) {
      const int64_t* RES pa = in[0];
      const int64_t* RES pb = in[1];
      const int64_t va = A == Val ? *pa : 0, vb = B == Val ? *pb : 0;
      const int64_t* const p[] = {in[I + 2]...};
      const int64_t v[] = {(Steps::form == Val ? *in[I + 2] : 0)...};
      int64_t* const m[] = {(Steps::storePrev ? mid[I] : nullptr)...};
      typename ArithOp<Op0>::T op0;
      for (pos_t i = 0; i < n; i++) {
         pos_t s = i;
         if constexpr (anySel) s = sel[i];
         int64_t acc = op0(at<A>(pa, va, i, s), at<B>(pb, vb, i, s));
         ((acc = chainStep<Steps>(acc, p[I], v[I], m[I], i, s)), ...);
         out[i] = acc;
      }
      return n;
   }
};
template <Arith Op0, Form A, Form B, typename... Steps>
pos_t chainKernel(pos_t n, const pos_t* sel, int64_t* const* mid, int64_t* out,
                  const int64_t* const* in) {
   return Chain<Op0, A, B, Steps...>::run(
       n, sel, mid, out, in, std::index_sequence_for<Steps...>{});
}
#endif

} // namespace compound
} // namespace primitives
} // namespace vectorwise
