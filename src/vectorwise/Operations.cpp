#include "vectorwise/Operations.hpp"

namespace vectorwise {


pos_t Expression::evaluate(pos_t n) {
   pos_t found = n;
   for (auto& op : ops) { found = op->run(found); }
   return found;
}

void Expression::operator+=(std::unique_ptr<Expression> other) {
   for (auto& op : other->ops) { ops.push_back(move(op)); }
   other->ops.clear();
}
void Expression::operator+=(std::unique_ptr<Op>&& op) {
   ops.push_back(move(op));
}

void Aggregates::operator+=(std::unique_ptr<Expression> other) {
   for (auto& op : other->ops) { ops.push_back(move(op)); }
   other->ops.clear();
}
void Aggregates::operator+=(std::unique_ptr<Op>&& op) {
   ops.push_back(move(op));
}

pos_t Aggregates::evaluate(pos_t n) {
   // Option 2: op-outer, tuple-inner.
   // Each primitive loops over all n tuples before the next primitive runs.
   auto found = 0;
   for (auto& aggr : ops) found = aggr->run(n);
   return found;
}

#ifdef VW_AGGR_TUPLE_OUTER
pos_t Aggregates::evaluate_tuple_outer(pos_t n) {
   // Tuple-outer, op-inner aggregation.
   // For each tuple i, apply all aggregate primitives to htMatches[i] before
   // moving to tuple i+1. This improves cache locality when the HT is large.
   //
   // We extract raw function pointers and args from each Op, then call them
   // directly in a tight loop — avoiding virtual dispatch, std::function, and
   // std::experimental::apply overhead per tuple.

   struct AggrInfo {
      enum Kind { COL, SEL } kind;
      primitives::FAggr fn_col;
      primitives::FAggrSel fn_sel;
      void* RES * entries;
      void* param1;
      pos_t* sel;
      size_t offset;
      size_t elemSize;
   };

   auto nOps = ops.size();
   AggrInfo info[16];
   assert(nOps <= 16);

   for (size_t j = 0; j < nOps; ++j) {
      auto* op = ops[j].get();
      if (auto* a = dynamic_cast<FAggrOp*>(op)) {
         info[j].kind = AggrInfo::COL;
         info[j].entries = std::get<0>(a->args);
         info[j].param1 = std::get<1>(a->args);
         info[j].sel = nullptr;
         info[j].offset = std::get<2>(a->args);
         info[j].elemSize = a->elemSize;
         info[j].fn_col = a->rawFn;
         info[j].fn_sel = nullptr;
      } else if (auto* a = dynamic_cast<FAggrSelOp*>(op)) {
         info[j].kind = AggrInfo::SEL;
         info[j].entries = std::get<0>(a->args);
         info[j].sel = std::get<1>(a->args);
         info[j].param1 = std::get<2>(a->args);
         info[j].offset = std::get<3>(a->args);
         info[j].elemSize = 0;
         info[j].fn_col = nullptr;
         info[j].fn_sel = a->rawFn;
      } else {
         // Unknown op — fall back to virtual dispatch
         for (pos_t i = 0; i < n; ++i) {
            for (auto& aggr : ops) aggr->run(1);
            for (auto& aggr : ops) aggr->advance(1);
         }
         for (auto& aggr : ops) aggr->advance(-static_cast<ptrdiff_t>(n));
         return n;
      }
   }

   // Tight tuple-outer loop: for each tuple, load the HT entry once, then
   // apply all aggregates to that entry before moving to the next tuple.
   // All aggregate values are int64_t and use += (plus), so we inline directly.
   for (pos_t i = 0; i < n; ++i) {
      auto* entry = reinterpret_cast<char*>(info[0].entries[i]);
      for (size_t j = 0; j < nOps; ++j) {
         auto& c = info[j];
         auto* aggregate = reinterpret_cast<int64_t*>(entry + c.offset);
         if (c.kind == AggrInfo::COL) {
            auto* col = reinterpret_cast<int64_t*>(c.param1);
            *aggregate += *col;
            c.param1 = reinterpret_cast<void*>(col + 1);
         } else {
            auto* col = reinterpret_cast<int64_t*>(c.param1);
            *aggregate += col[c.sel[i]];
         }
      }
   }
   return n;
}
#endif // VW_AGGR_TUPLE_OUTER

Scatter::ScatterInfo::ScatterInfo(std::unique_ptr<Op>&& operation, void** s,
                                  size_t off)
    : op(move(operation)), start(s), offset(off) {}

pos_t Scatter::evaluate(pos_t n, void* start) {
   for (auto& scat : ops) {
      *scat.start = reinterpret_cast<uint8_t*>(start) + scat.offset;
      scat.op->run(n);
   }
   return n;
}

GatherOpCol::GatherOpCol(primitives::FGather o, void** s, size_t off, void* t)
    : op(o), source(s), offset(off), target(t) {}

pos_t GatherOpCol::run(pos_t n) { return op(n, source, offset, target); }

GatherOpVal::GatherOpVal(primitives::FGatherVal o, void** s, size_t off,
                         size_t* s_s, void* t)
    : op(o), sourceStart(s), offset(off), struct_size(s_s), target(t) {}

pos_t GatherOpVal::run(pos_t n) {
   return op(n, sourceStart, offset, struct_size, target);
}

EqualityCheck::EqualityCheck(primitives::EQCheck p, void** ptrs, size_t off,
                             pos_t* probeI, void* probeD)
    : prim(p), pointers(ptrs), offset(off), probeIdxs(probeI),
      probeData(probeD) {}

pos_t EqualityCheck::run(pos_t n) {
   return prim(n, pointers, offset, probeIdxs, probeData);
}
NEqualityCheck::NEqualityCheck(primitives::NEQCheck e, pos_t* en, void** entr,
                               void* p, size_t off, SizeBuffer<pos_t>* nEq)
    : eq(e), entryIdx(en), entry(entr), probeKey(p), offset(off), notEq(nEq) {}

pos_t NEqualityCheck::run(pos_t n) {
   return eq(n, entryIdx, entry, probeKey, offset, notEq);
}

NEqualityCheckSel::NEqualityCheckSel(primitives::NEQCheckSel e, pos_t* idx,
                                     void** entr, pos_t* prob, void* probeK,
                                     size_t off, SizeBuffer<pos_t>* nEq)
    : eq(e), entryIdx(idx), entry(entr), probeSel(prob), probeKey(probeK),
      offset(off), notEq(nEq){};

pos_t NEqualityCheckSel::run(pos_t n) {
   return eq(n, entryIdx, entry, probeSel, probeKey, offset, notEq);
}

pos_t F1_Op::run(pos_t n) { return operation(n, input); }
pos_t F2_Op::run(pos_t n) { return operation(n, input, param1); }
pos_t F3_Op::run(pos_t n) {
   return operation(n, outputSelectionV, param1, param2);
}
pos_t F4_Op::run(pos_t n) {
   return operation(n, inputSelectionV, outputSelectionV, param1, param2);
}
}
