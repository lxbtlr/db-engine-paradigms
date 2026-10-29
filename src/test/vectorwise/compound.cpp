// VW_PROJ_COMPOUND: differential tests for the compound projection
// primitives. Every descriptor must match what its primitive computes, every
// canonical kernel (and every chain kernel the build lists) must match
// separate scalar steps, and ExpressionBuilder must fuse exactly the chains
// it may.
#ifdef VW_PROJ_COMPOUND
#include "vectorwise/CompoundPrimitives.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using namespace vectorwise::primitives::compound;

namespace {

constexpr pos_t kRows = 1000;

struct Data {
   std::vector<int64_t> x, y, z;
   std::vector<pos_t> sel;
   int64_t vx = 7, vy = -3, vz = 11;
   Data() {
      std::mt19937_64 rng(1);
      for (pos_t i = 0; i < kRows; ++i) {
         x.push_back(int64_t(rng() % 2001) - 1000);
         y.push_back(int64_t(rng() % 2001) - 1000);
         z.push_back(int64_t(rng() % 2001) - 1000);
         if (rng() % 10 < 8) sel.push_back(i); // ~80% pass, ascending
      }
   }
};

int64_t apply(Arith op, int64_t l, int64_t r) {
   return op == Plus ? l + r : op == Minus ? l - r : l * r;
}
int64_t read(Form f, const std::vector<int64_t>& col, int64_t v,
             const std::vector<pos_t>& sel, pos_t i) {
   return f == Col ? col[i] : f == Sel ? col[sel[i]] : v;
}
const int64_t* arg(Form f, const std::vector<int64_t>& col, const int64_t& v) {
   return f == Val ? &v : col.data();
}

struct Named {
   const void* prim;
   Arith op;
   Form a, b;
   bool sel;
};
#define NAMED(op, A)                                                           \
   {(const void*)proj_##op##_int64_t_col_int64_t_col, A, Col, Col, false},     \
       {(const void*)proj_##op##_int64_t_col_int64_t_val, A, Col, Val, false}, \
       {(const void*)proj_sel_both_##op##_int64_t_col_int64_t_col, A, Sel,     \
        Sel, true},                                                            \
       {(const void*)proj_##op##_sel_int64_t_col_int64_t_col, A, Sel, Col,     \
        true},                                                                 \
       {(const void*)proj_##op##_int64_t_col_sel_int64_t_col, A, Col, Sel,     \
        true},                                                                 \
       {(const void*)proj_sel_##op##_int64_t_col_int64_t_val, A, Sel, Val,     \
        true},

} // namespace

TEST(Compound, DescriptorsMatchPrimitives) {
   const Named named[] = {
       NAMED(plus, Plus) NAMED(minus, Minus) NAMED(multiplies, Mul)
       {(const void*)proj_minus_int64_t_val_int64_t_col, Minus, Val, Col, false},
       {(const void*)proj_sel_minus_int64_t_val_int64_t_col, Minus, Val, Sel,
        true},
   };
   Data d;
   for (auto& p : named) {
      Desc desc;
      ASSERT_TRUE(describe(p.prim, desc));
      EXPECT_EQ(desc.op, p.op);
      EXPECT_EQ(desc.a, p.a);
      EXPECT_EQ(desc.b, p.b);
      EXPECT_EQ(desc.selInput, p.sel);
      // run the primitive itself and compare with what the Desc says
      const pos_t n = desc.selInput ? pos_t(d.sel.size()) : kRows;
      std::vector<int64_t> out(kRows);
      void* a = (void*)arg(desc.a, d.x, d.vx);
      void* b = (void*)arg(desc.b, d.y, d.vy);
      if (desc.selInput)
         ((F4)p.prim)(n, d.sel.data(), out.data(), a, b);
      else
         ((F3)p.prim)(n, out.data(), a, b);
      for (pos_t i = 0; i < n; ++i)
         ASSERT_EQ(out[i], apply(desc.op, read(desc.a, d.x, d.vx, d.sel, i),
                                 read(desc.b, d.y, d.vy, d.sel, i)))
             << "primitive " << &p - named << " row " << i;
   }
   Desc none;
   EXPECT_FALSE(describe((const void*)proj_plus_int32_t_col_int32_t_col, none));
   EXPECT_FALSE(describe((const void*)proj_divides_int64_t_col_int64_t_col, none));
}

TEST(Compound, KernelsMatchTwoSteps) {
   Data d;
   size_t checked = 0;
   for (int op1 = 0; op1 < 3; ++op1)
      for (int a = 0; a < 3; ++a)
         for (int b = 0; b < 3; ++b)
            for (int op2 = 0; op2 < 3; ++op2)
               for (int c = 0; c < 3; ++c)
                  for (int cLeft = 0; cLeft < 2; ++cLeft)
                  for (int storeMid = 0; storeMid < 2; ++storeMid) {
                     const bool canonical =
                         !(a == Val && b == Val) &&
                         !(commutative(Arith(op1)) && a > b) &&
                         !(commutative(Arith(op2)) && cLeft);
                     Fn k = kernel(Arith(op1), Form(a), Form(b), Arith(op2),
                                   Form(c), cLeft, storeMid);
                     ASSERT_EQ(k != nullptr, canonical);
                     if (!k) continue;
                     const bool anySel = a == Sel || b == Sel || c == Sel;
                     const pos_t n = anySel ? pos_t(d.sel.size()) : kRows;
                     std::vector<int64_t> mid(kRows, -7), out(kRows);
                     k(n, d.sel.data(), mid.data(), out.data(),
                       arg(Form(a), d.x, d.vx), arg(Form(b), d.y, d.vy),
                       arg(Form(c), d.z, d.vz));
                     for (pos_t i = 0; i < n; ++i) {
                        const int64_t m =
                            apply(Arith(op1), read(Form(a), d.x, d.vx, d.sel, i),
                                  read(Form(b), d.y, d.vy, d.sel, i));
                        const int64_t z = read(Form(c), d.z, d.vz, d.sel, i);
                        ASSERT_EQ(mid[i], storeMid ? m : -7); // -7: untouched
                        ASSERT_EQ(out[i], cLeft ? apply(Arith(op2), z, m)
                                                : apply(Arith(op2), m, z));
                     }
                     ++checked;
                  }
   EXPECT_EQ(checked, 432u);
}

namespace {
// the builder types are protected: re-export them
struct Exposed : QueryBuilder {
   using QueryBuilder::DS;
   using QueryBuilder::ExpressionBuilder;
};
using DS = Exposed::DS;
DS buffer(void* p) {
   DS r;
   r.buf = DS::BufferSpec::Buffer;
   r.dataSize = sizeof(int64_t);
   r.data = p;
   return r;
}
DS value(int64_t* p) {
   DS r;
   r.buf = DS::BufferSpec::Value;
   r.data = p;
   return r;
}
DS scratch(void* p) {
   DS r = buffer(p);
   r.scratch = true;
   return r;
}
/// a pair (CompoundOp) or, with VW_COMPOUND_CHAINS, a chain (ChainOp)
bool fusedOp(Op* op) {
#ifdef VW_COMPOUND_CHAINS
   if (dynamic_cast<ChainOp*>(op)) return true;
#endif
   return dynamic_cast<CompoundOp*>(op) != nullptr;
}
size_t inRegister(Op* op) {
#ifdef VW_COMPOUND_CHAINS
   if (auto* c = dynamic_cast<ChainOp*>(op)) return c->inRegister;
#endif
   auto* c = dynamic_cast<CompoundOp*>(op);
   return c && !c->storeMid;
}
Exposed::ExpressionBuilder builder(std::unordered_set<void*>* elided = nullptr) {
   Exposed::ExpressionBuilder b;
   b.expression = std::make_unique<Expression>();
   b.elided = elided;
   return b;
}
} // namespace

TEST(Compound, BuilderFusesQ1Chain) {
   // Q1: disc_price = price[sel] * (one - disc[sel]); charge needs a second
   // expression and is covered by the same path
   Data d;
   int64_t one = 100;
   std::vector<int64_t> minus(kRows), discPrice(kRows);
   auto b = builder();
   b.addOp(proj_sel_minus_int64_t_val_int64_t_col, buffer(d.sel.data()),
           buffer(minus.data()), value(&one), buffer(d.x.data()))
       .addOp(proj_multiplies_sel_int64_t_col_int64_t_col, buffer(d.sel.data()),
              buffer(discPrice.data()), buffer(d.y.data()),
              buffer(minus.data()));
   std::unique_ptr<Expression> e = b;
   ASSERT_EQ(e->ops.size(), 1u);
   ASSERT_TRUE(fusedOp(e->ops[0].get()));
   const pos_t n = pos_t(d.sel.size());
   EXPECT_EQ(e->evaluate(n), n);
   for (pos_t i = 0; i < n; ++i) {
      ASSERT_EQ(minus[i], one - d.x[d.sel[i]]);
      ASSERT_EQ(discPrice[i], d.y[d.sel[i]] * minus[i]);
   }
}

TEST(Compound, BuilderLeavesUnchainedAndAliasedOps) {
   Data d;
   int64_t one = 100;
   std::vector<int64_t> t1(kRows), t2(kRows);
   // independent: the second does not read the first's result
   auto b = builder();
   b.addOp(proj_plus_int64_t_col_int64_t_val, buffer(t1.data()),
           buffer(d.x.data()), value(&one))
       .addOp(proj_plus_int64_t_col_int64_t_val, buffer(t2.data()),
              buffer(d.y.data()), value(&one));
   std::unique_ptr<Expression> e = b;
   ASSERT_EQ(e->ops.size(), 2u);
   EXPECT_FALSE(fusedOp(e->ops[0].get()));
   // aliased: the second writes over the first's input
   std::vector<int64_t> in(d.x);
   auto b2 = builder();
   b2.addOp(proj_plus_int64_t_col_int64_t_val, buffer(t1.data()),
            buffer(in.data()), value(&one))
       .addOp(proj_multiplies_int64_t_col_int64_t_col, buffer(in.data()),
              buffer(t1.data()), buffer(d.y.data()));
   std::unique_ptr<Expression> e2 = b2;
   ASSERT_EQ(e2->ops.size(), 2u);
   e2->evaluate(kRows);
   for (pos_t i = 0; i < kRows; ++i)
      ASSERT_EQ(in[i], (d.x[i] + one) * d.y[i]);
   // three chained ops: pairs, so the third stays a primitive
   std::vector<int64_t> t3(kRows);
   auto b3 = builder();
   b3.addOp(proj_plus_int64_t_col_int64_t_val, buffer(t1.data()),
            buffer(d.x.data()), value(&one))
       .addOp(proj_multiplies_int64_t_col_int64_t_col, buffer(t2.data()),
              buffer(t1.data()), buffer(d.y.data()))
       .addOp(proj_minus_int64_t_col_int64_t_col, buffer(t3.data()),
              buffer(t2.data()), buffer(d.z.data()));
   std::unique_ptr<Expression> e3 = b3;
#ifndef VW_COMPOUND_CHAINS
   ASSERT_EQ(e3->ops.size(), 2u);
   EXPECT_NE(dynamic_cast<CompoundOp*>(e3->ops[0].get()), nullptr);
#endif
   e3->evaluate(kRows);
   for (pos_t i = 0; i < kRows; ++i)
      ASSERT_EQ(t3[i], (d.x[i] + one) * d.y[i] - d.z[i]);
}

TEST(Compound, ScratchMidStaysInRegister) {
   Data d;
   int64_t c = 3;
   std::vector<int64_t> s(kRows, -7), out(kRows), late(kRows);
   std::unordered_set<void*> elided;
   // out = (x * c) + y[sel]: the Scratch product is never stored
   auto b = builder(&elided);
   b.addOp(proj_multiplies_int64_t_col_int64_t_val, scratch(s.data()),
           buffer(d.x.data()), value(&c))
       .addOp(proj_plus_int64_t_col_sel_int64_t_col, buffer(d.sel.data()),
              buffer(out.data()), scratch(s.data()), buffer(d.y.data()));
   std::unique_ptr<Expression> e = b;
   ASSERT_EQ(e->ops.size(), 1u);
   ASSERT_TRUE(fusedOp(e->ops[0].get()));
   EXPECT_EQ(inRegister(e->ops[0].get()), 1u);
   const pos_t n = pos_t(d.sel.size());
   e->evaluate(n);
   for (pos_t i = 0; i < n; ++i) {
      ASSERT_EQ(out[i], d.x[i] * c + d.y[d.sel[i]]);
      ASSERT_EQ(s[i], -7);
   }
   // a later read of the elided Scratch is refused at build time
   auto b2 = builder(&elided);
   EXPECT_THROW(b2.addOp(proj_plus_int64_t_col_int64_t_col, buffer(late.data()),
                         scratch(s.data()), buffer(d.x.data()))
                    .addOp(proj_plus_int64_t_col_int64_t_val, buffer(out.data()),
                           buffer(d.y.data()), value(&c)),
                std::runtime_error);
   // once rewritten by an unfused op it is readable again
   auto b3 = builder(&elided);
   b3.addOp(proj_plus_int64_t_col_int64_t_val, scratch(s.data()),
            buffer(d.x.data()), value(&c));
   std::unique_ptr<Expression> e3 = b3;
   auto b4 = builder(&elided);
   EXPECT_NO_THROW(b4.addOp(proj_plus_int64_t_col_int64_t_col,
                            buffer(late.data()), scratch(s.data()),
                            buffer(d.y.data()))
                       .addOp(proj_plus_int64_t_col_int64_t_val,
                              buffer(out.data()), buffer(d.z.data()),
                              value(&c)));
}

#ifdef VW_COMPOUND_CHAINS
namespace {
ChainSig parse(const std::string& s) {
   auto op = [](const std::string& t) { return t == "plus" ? Plus : t == "minus" ? Minus : Mul; };
   auto form = [](const std::string& t) { return t == "col" ? Col : t == "sel" ? Sel : Val; };
   std::istringstream in(s);
   std::string o, a, b, sep, keep, side;
   in >> o >> a >> b;
   ChainSig sig{op(o), form(a), form(b), {}};
   while (in >> sep >> keep >> o >> a >> side)
      sig.steps.push_back({op(o), form(a), side == "l", keep == "keep"});
   return sig;
}
} // namespace

TEST(Compound, ChainKernelsMatchSteps) {
   Data d;
   std::vector<int64_t> v(64, 5);
   for (size_t e = 0; e < chainEntryCount; ++e) {
      const ChainSig sig = parse(chainEntries[e].sig);
      ASSERT_EQ(sig.str(), chainEntries[e].sig);
      ASSERT_EQ(findChain(sig), chainEntries[e].fn);
      // operand k: column x, y or z by k % 3, or a constant
      const std::vector<int64_t>* cols[] = {&d.x, &d.y, &d.z};
      std::vector<Form> forms = {sig.a, sig.b};
      for (auto& s : sig.steps) forms.push_back(s.form);
      std::vector<const int64_t*> in;
      for (size_t k = 0; k < forms.size(); ++k) {
         v[k] = int64_t(k) * 3 - 4;
         in.push_back(arg(forms[k], *cols[k % 3], v[k]));
      }
      bool anySel = false;
      for (auto f : forms) anySel |= f == Sel;
      const pos_t n = anySel ? pos_t(d.sel.size()) : kRows;
      std::vector<std::vector<int64_t>> mids(sig.steps.size(), std::vector<int64_t>(kRows, -7));
      std::vector<int64_t*> mid;
      for (auto& m : mids) mid.push_back(m.data());
      std::vector<int64_t> out(kRows);
      chainEntries[e].fn(n, d.sel.data(), mid.data(), out.data(), in.data());
      for (pos_t i = 0; i < n; ++i) {
         auto rd = [&](size_t k) { return read(forms[k], *cols[k % 3], v[k], d.sel, i); };
         int64_t acc = apply(sig.op0, rd(0), rd(1));
         for (size_t s = 0; s < sig.steps.size(); ++s) {
            ASSERT_EQ(mids[s][i], sig.steps[s].storePrev ? acc : -7) << chainEntries[e].sig;
            const int64_t x = rd(s + 2);
            acc = sig.steps[s].left ? apply(sig.steps[s].op, x, acc) : apply(sig.steps[s].op, acc, x);
         }
         ASSERT_EQ(out[i], acc) << chainEntries[e].sig << " row " << i;
      }
   }
}

TEST(Compound, BuilderFusesChainsUpToDepth) {
   // out = ((x * c) + y[sel]) - z through two Scratch intermediates: one
   // ChainOp if this build has the three-step kernel and the depth allows
   // it, else the longest compiled prefix and a primitive
   Data d;
   int64_t c = 3;
   std::vector<int64_t> s1(kRows, -7), s2(kRows, -7), out(kRows);
   std::unordered_set<void*> elided;
   auto b = builder(&elided);
   b.addOp(proj_multiplies_int64_t_col_int64_t_val, scratch(s1.data()),
           buffer(d.x.data()), value(&c))
       .addOp(proj_plus_int64_t_col_sel_int64_t_col, buffer(d.sel.data()),
              scratch(s2.data()), scratch(s1.data()), buffer(d.y.data()))
       .addOp(proj_minus_int64_t_col_int64_t_col, buffer(out.data()),
              scratch(s2.data()), buffer(d.z.data()));
   std::unique_ptr<Expression> e = b;
   const ChainSig three{Mul, Col, Val, {{Plus, Sel, false, false}, {Minus, Col, false, false}}};
   const ChainSig two{Mul, Col, Val, {{Plus, Sel, false, false}}};
   if (VW_COMPOUND_DEPTH >= 3 && findChain(three)) {
      ASSERT_EQ(e->ops.size(), 1u);
      EXPECT_EQ(inRegister(e->ops[0].get()), 2u);
   } else if (findChain(two)) {
      ASSERT_EQ(e->ops.size(), 2u);
      EXPECT_EQ(inRegister(e->ops[0].get()), 1u);
   } else {
      ASSERT_EQ(e->ops.size(), 3u);
   }
   const pos_t n = pos_t(d.sel.size());
   e->evaluate(n);
   for (pos_t i = 0; i < n; ++i) {
      ASSERT_EQ(out[i], d.x[i] * c + d.y[d.sel[i]] - d.z[i]);
      ASSERT_EQ(s1[i], -7);
   }
}
#endif
#endif
