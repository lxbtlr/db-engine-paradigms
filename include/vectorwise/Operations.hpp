#pragma once
#include "Primitives.hpp"
#include <experimental/tuple>
#include <functional>
#include <memory>
#include <vector>

namespace vectorwise {

// TODO: Put a class hierarchy into functions: Operation, Aggregate, Scatter,
// Gather

struct Op;
class Expression {
 public:
   std::vector<std::unique_ptr<Op>> ops;
   /// Evaluate all operations of this expression
   pos_t evaluate(pos_t n);
   void operator+=(std::unique_ptr<Expression> other);
   void operator+=(std::unique_ptr<Op>&& op);
};

class Aggregates {
 public:
   std::vector<std::unique_ptr<Op>> ops;
   /// Option 2 (default): op-outer, tuple-inner.
   /// Calls each primitive once with the full batch of n tuples.
   pos_t evaluate(pos_t n);
#ifdef VW_AGGR_TUPLE_OUTER
   /// Option 1: tuple-outer, op-inner.
   /// Iterates one tuple at a time; for each tuple calls every primitive
   /// with n=1 so that all ops are applied to a single row before moving on.
   pos_t evaluate_tuple_outer(pos_t n);
#endif
   void operator+=(std::unique_ptr<Expression> other);
   void operator+=(std::unique_ptr<Op>&& op);
};

class Scatter {
   struct ScatterInfo {
      std::unique_ptr<Op> op;
      void** start;
      size_t offset;
      ScatterInfo(std::unique_ptr<Op>&& operation, void** s, size_t off);
   };

 public:
   std::vector<ScatterInfo> ops;
   /// run scatter operations to spread to memory region pointed to by start
   pos_t evaluate(pos_t n, void* start);
};

struct Op {
   virtual pos_t run(pos_t n) = 0;
#ifdef VW_AGGR_TUPLE_OUTER
   /// Advance all data pointers by `step` elements (element size is
   /// op-specific).  Used by Aggregates::evaluate_tuple_outer to step through
   /// one tuple at a time.  Negative step rewinds.  Default is a no-op; ops
   /// that participate in tuple-outer aggregation must override this.
   virtual void advance(ptrdiff_t /*step*/) {}
#endif
   virtual ~Op() = default;
};

template <typename> class OpArgs;

template <typename... Args>
class OpArgs<pos_t (*)(pos_t, Args...)> : public Op {
   std::function<pos_t(pos_t, Args...)> function;

 public:
   std::tuple<Args...> args;

   OpArgs(pos_t(f)(pos_t, Args...), Args... x) : function(f), args(x...) {}

   template <unsigned n> decltype(std::get<n>(args)) get() {
      return std::get<n>(args);
   }

   virtual pos_t run(pos_t n) override {
      return std::experimental::apply(function,
                                      std::tuple_cat(std::make_tuple(n), args));
   }
};

using FScatterOp = OpArgs<primitives::FScatter>;
using FScatterSelOp = OpArgs<primitives::FScatterSel>;
using FScatterSelRowOp = OpArgs<primitives::FScatterSelRow>;
using FAggrRowOp = OpArgs<primitives::FAggrRow>;
using FAggrInitOp = OpArgs<primitives::FAggrInit>;

#ifdef VW_AGGR_TUPLE_OUTER
// FAggrOp and FAggrSelOp are defined as named structs (not OpArgs aliases)
// so they can carry elemSize and override advance() for tuple-outer evaluation.
// FAggr signature: pos_t(pos_t n, void* entries[], void* param1, size_t offset)
//   arg 0: entries (void**) — htMatches array, advances by sizeof(void*)/element
//   arg 1: param1  (void*)  — column data, advances by elemSize bytes
//   arg 2: offset  (size_t) — constant
struct FAggrOp : public Op {
   primitives::FAggr function;
   void** entries;   // htMatches
   void*  param1;    // column data
   size_t offset;
   size_t elemSize;  // byte size of one element in param1 column

   FAggrOp(primitives::FAggr f, void** e, void* p, size_t off, size_t esz)
       : function(f), entries(e), param1(p), offset(off), elemSize(esz) {}
   // Convenience ctor without elemSize (defaults 0; advance() is no-op)
   FAggrOp(primitives::FAggr f, void** e, void* p, size_t off)
       : FAggrOp(f, e, p, off, 0) {}

   // Required by existing QueryBuilder code that calls get<N>() on the op.
   // Map indices to fields to match original OpArgs<FAggr> layout.
   template <unsigned N> auto& get();

   virtual pos_t run(pos_t n) override {
      return function(n, entries, param1, offset);
   }
   virtual void advance(ptrdiff_t step) override {
      entries += step;
      param1 = addBytes(param1, step * static_cast<ptrdiff_t>(elemSize));
   }
};

template <> inline auto& FAggrOp::get<0>() { return entries; }
template <> inline auto& FAggrOp::get<1>() { return param1; }
template <> inline auto& FAggrOp::get<2>() { return offset; }

// FAggrSel signature: pos_t(pos_t n, void* entries[], pos_t* sel, void* param1, size_t offset)
//   arg 0: entries (void**) — advances by 1 pointer per step
//   arg 1: sel     (pos_t*) — selection vector into param1, advances by 1
//   arg 2: param1  (void*)  — column data base, does NOT advance (sel does the indirection)
//   arg 3: offset  (size_t) — constant
struct FAggrSelOp : public Op {
   primitives::FAggrSel function;
   void**  entries;
   pos_t*  sel;
   void*   param1;
   size_t  offset;

   FAggrSelOp(primitives::FAggrSel f, void** e, pos_t* s, void* p, size_t off)
       : function(f), entries(e), sel(s), param1(p), offset(off) {}

   template <unsigned N> auto& get();

   virtual pos_t run(pos_t n) override {
      return function(n, entries, sel, param1, offset);
   }
   virtual void advance(ptrdiff_t step) override {
      entries += step;
      sel     += step;
      // param1 is indexed via sel, so its base pointer stays fixed
   }
};

template <> inline auto& FAggrSelOp::get<0>() { return entries; }
template <> inline auto& FAggrSelOp::get<1>() { return sel; }
template <> inline auto& FAggrSelOp::get<2>() { return param1; }
template <> inline auto& FAggrSelOp::get<3>() { return offset; }
#else
// Without tuple-outer evaluation, FAggrOp and FAggrSelOp are simple OpArgs aliases.
using FAggrOp    = OpArgs<primitives::FAggr>;
using FAggrSelOp = OpArgs<primitives::FAggrSel>;
#endif // VW_AGGR_TUPLE_OUTER

using FPartitionByKeyOp = OpArgs<primitives::FPartitionByKey>;
using FPartitionByKeySelOp = OpArgs<primitives::FPartitionByKeySel>;
using FPartitionByKeyRowOp = OpArgs<primitives::FPartitionByKeyRow>;
using NEQCheckRowOp = OpArgs<primitives::NEQCheckRow>;
using F6_Op = OpArgs<primitives::F6>;
using F7_Op = OpArgs<primitives::F7>;

struct GatherOpCol : public Op {
   primitives::FGather op;
   void** source;
   size_t offset;
   void* target;
   GatherOpCol(primitives::FGather op, void** source, size_t off, void* target);
   virtual pos_t run(pos_t n) override;
};

struct GatherOpVal : public Op {
   primitives::FGatherVal op;
   void** sourceStart;
   size_t offset;
   size_t* struct_size;
   void* target;
   GatherOpVal(primitives::FGatherVal op, void** source, size_t off,
               size_t* struct_size, void* target);
   virtual pos_t run(pos_t n) override;
};

struct EqualityCheck : public Op {
   primitives::EQCheck prim;
   void** pointers;
   size_t offset;
   pos_t* probeIdxs;
   void* probeData;
   EqualityCheck(primitives::EQCheck p, void** ptrs, size_t off,
                 pos_t* probeIdxs, void* probeData);
   /// run check operations
   virtual pos_t run(pos_t n) override;
};

struct NEqualityCheck : public Op {
   primitives::NEQCheck eq;
   pos_t* entryIdx;
   void** entry;
   void* probeKey;
   size_t offset;
   SizeBuffer<pos_t>* notEq;
   NEqualityCheck(primitives::NEQCheck eq, pos_t* entryIdx, void** entry,
                  void* probeKey, size_t offset, SizeBuffer<pos_t>* notEq);
   virtual pos_t run(pos_t n) override;
};

struct NEqualityCheckSel : public Op {
   primitives::NEQCheckSel eq;
   pos_t* entryIdx;
   void** entry;
   pos_t* probeSel;
   void* probeKey;
   size_t offset;
   SizeBuffer<pos_t>* notEq;
   NEqualityCheckSel(primitives::NEQCheckSel eq, pos_t* entryIdx, void** entry,
                     pos_t* probes, void* probeKey, size_t offset,
                     SizeBuffer<pos_t>* notEq);
   virtual pos_t run(pos_t n) override;
};

struct F1_Op : public Op {
   void* input;
   primitives::F1 operation;
   F1_Op(void* i, primitives::F1 op) : input(i), operation(op) {}
   virtual pos_t run(pos_t n) override;
};

struct F2_Op : public Op {
   void* input;
   void* param1;
   primitives::F2 operation;
   F2_Op(void* i, void* p1, primitives::F2 op)
       : input(i), param1(p1), operation(op) {}
   virtual pos_t run(pos_t n) override;
};

struct F3_Op : public Op
/// selection with two params
{
   void* outputSelectionV;
   void* param1;
   void* param2;
   primitives::F3 operation;
   F3_Op(void* out, void* p1, void* p2, primitives::F3 o)
       : outputSelectionV(out), param1(p1), param2(p2), operation(o) {}
   virtual pos_t run(pos_t n) override;
};

struct F4_Op : public Op
/// selection with input select and two params
{
   void* inputSelectionV;
   void* outputSelectionV;
   void* param1;
   void* param2;
   primitives::F4 operation;
   F4_Op(void* in, void* out, void* p1, void* p2, primitives::F4 o)
       : inputSelectionV(in), outputSelectionV(out), param1(p1), param2(p2),
         operation(o) {}
   virtual pos_t run(pos_t n) override;
};
}
