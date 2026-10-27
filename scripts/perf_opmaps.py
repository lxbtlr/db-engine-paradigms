"""Operator labels for scripts/perf_ops_report.py.

TECTORWISE[query]: (path, label) pairs. A path is perf_opfold.py's operator
key without the leading "ResultWriter > " (every plan's root), and it must
match the whole key (re.fullmatch). The trees follow the query builders in
src/benchmarks/tpch/queries/qN.cpp: QueryBuilder is a stack machine, and
HashJoin pops its probe side (the top) and its build side (below it).
Scans are labelled with their table; the first label that matches wins.

TYPER[query]: {"pipelines": [(regex on the pipeline's call-site source,
name)], "operators": [(regex on "<source line> [<inlined function>]", label)]}.
The source text identifies a line across commits where line numbers move.
"""
import re

HJ, HG = "Hashjoin", "HashGroup"


def _tw(pairs):
    out = []
    for path, label in pairs:
        out.append(("ResultWriter > " + re.escape(path), label))
    out.append(("ResultWriter", "result writer"))
    return out


TECTORWISE = {
    1: _tw([
        (f"{HG}[global]", "Γ global + output"),
        (f"{HG}[local]", "Γ local (l_returnflag, l_linestatus)"),
        (f"{HG}[local] > Project", "π charge, disc_price"),
        (f"{HG}[local] > Project > Select", "σ l_shipdate"),
        (f"{HG}[local] > Project > Select > Scan", "scan lineitem"),
    ]),
    3: _tw([
        (f"{HG}[global]", "Γ global + output"),
        (f"{HG}[local]", "Γ local (l_orderkey, o_orderdate, o_shippriority)"),
        (f"{HG}[local] > Project", "π revenue"),
        (f"{HG}[local] > Project > {HJ}[probe]", "⋈ orders-lineitem: probe"),
        (f"{HG}[local] > Project > {HJ}[probe] > Select", "σ l_shipdate"),
        (f"{HG}[local] > Project > {HJ}[probe] > Select > Scan", "scan lineitem"),
        (f"{HG}[local] > Project > {HJ}[build]", "⋈ orders-lineitem: build"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[probe]", "⋈ customer-orders: probe"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[probe] > Select", "σ o_orderdate"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[probe] > Select > Scan", "scan orders"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build]", "⋈ customer-orders: build"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > Select", "σ c_mktsegment"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > Select > Scan", "scan customer"),
    ]),
    5: _tw([
        (f"{HG}[global]", "Γ global + output"),
        (f"{HG}[local]", "Γ local (n_name)"),
        (f"{HG}[local] > Project", "π revenue"),
        # supplier join: build supplier, probe the lineitem matches
        (f"{HG}[local] > Project > {HJ}[probe]", "⋈ supplier: probe"),
        (f"{HG}[local] > Project > {HJ}[build]", "⋈ supplier: build"),
        (f"{HG}[local] > Project > {HJ}[build] > Scan", "scan supplier"),
        # orders-lineitem: build the orders matches, probe lineitem
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[probe]", "⋈ orders-lineitem: probe"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[probe] > Scan", "scan lineitem"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build]", "⋈ orders-lineitem: build"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[probe]", "⋈ customer-orders: probe"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[probe] > Select", "σ o_orderdate"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[probe] > Select > Scan", "scan orders"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[build]", "⋈ customer-orders: build"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[build] > {HJ}[probe]",
         "⋈ nation-customer: probe"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[build] > {HJ}[probe] > Scan", "scan customer"),
        (f"{HG}[local] > Project > {HJ}[probe] > {HJ}[build] > {HJ}[build] > {HJ}[build]",
         "⋈ nation-customer: build"),
    ]),
    6: _tw([
        ("FixedAggr[output]", "Γ output"),
        ("FixedAggr[consume]", "Γ sum (fixed)"),
        ("FixedAggr[consume] > Project", "π l_extendedprice * l_discount"),
        ("FixedAggr[consume] > Project > Select", "σ shipdate, discount, quantity"),
        ("FixedAggr[consume] > Project > Select > Scan", "scan lineitem"),
    ]),
    9: _tw([
        (f"{HG}[global]", "Γ global + output"),
        (f"{HG}[local]", "Γ local (nation, o_year)"),
        (f"{HG}[local] > Project", "π amount"),
        # last join: build the lineitem matches, probe orders
        (f"{HG}[local] > Project > {HJ}[probe]", "⋈ lineitem-orders: probe"),
        (f"{HG}[local] > Project > {HJ}[probe] > Scan", "scan orders"),
        (f"{HG}[local] > Project > {HJ}[build]", "⋈ lineitem-orders: build"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[probe]", "⋈ partsupp-lineitem: probe"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[probe] > Scan", "scan lineitem"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build]", "⋈ partsupp-lineitem: build"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe]", "⋈ supplier-partsupp: probe"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe] > {HJ}[probe]",
         "⋈ part-partsupp (semi): probe"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe] > {HJ}[probe] > Scan", "scan partsupp"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe] > {HJ}[build]",
         "⋈ part-partsupp (semi): build"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select", "σ p_name"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select > Scan",
         "scan part"),
        (f"{HG}[local] > Project > {HJ}[build] > {HJ}[build] > {HJ}[build]", "⋈ supplier-partsupp: build"),
    ]),
    18: _tw([
        (f"{HG}[global]", "Γ global + output"),
        (f"{HG}[local]", "Γ local (c_name ... o_totalprice)"),
        (f"{HG}[local] > {HJ}[probe]", "⋈ orders-lineitem: probe"),
        (f"{HG}[local] > {HJ}[probe] > Scan", "scan lineitem (outer)"),
        (f"{HG}[local] > {HJ}[build]", "⋈ orders-lineitem: build"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe]", "⋈ customer-orders: probe"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[build]", "⋈ customer-orders: build"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[build] > Scan", "scan customer"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[probe]", "⋈ semi o_orderkey in subquery: probe"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[probe] > Scan", "scan orders"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[build]", "⋈ semi o_orderkey in subquery: build"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select", "σ sum(l_quantity) > 300"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select > {HG}[global]",
         "subquery Γ global (l_orderkey)"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select > {HG}[local]",
         "subquery Γ local (l_orderkey)"),
        (f"{HG}[local] > {HJ}[build] > {HJ}[probe] > {HJ}[build] > Select > {HG}[local] > Scan",
         "scan lineitem (subquery)"),
    ]),
}

SCAN_LOOP = (r"^for \(size_t i = r\.begin\(\)", "scan loop")

TYPER = {
    1: {
        "pipelines": [
            (r"^tbb::parallel_for\(", "P1 lineitem: σ, Γ local"),
            (r"forallGroups", "P2 Γ global + output"),
            (r"initQuery", "setup"),
        ],
        "operators": [
            (r"getGroup|make_tuple\(l_returnflag", "Γ local: group lookup"),
            (r"get<\d>\(group\) \+=", "Γ local: aggregate update"),
            (r"auto (disc_price|charge) =", "π disc_price, charge"),
            (r"l_shipdate\[i\] <= c1", "σ l_shipdate"),
            (r"forallGroups", "Γ global + output"),
            SCAN_LOOP,
        ],
    },
    3: {
        "pipelines": [
            (r"found1 = tbb::parallel_reduce", "P1 customer: σ, ht1 entries"),
            (r"parallel_insert\(entries1|ht1\.setSize", "P1 ht1 insert"),
            (r"found2 = tbb::parallel_reduce", "P2 orders: σ, probe ht1, ht2 entries"),
            (r"parallel_insert\(entries2|ht2\.setSize", "P2 ht2 insert"),
            (r"^tbb::parallel_for\(", "P3 lineitem: σ, probe ht2, Γ local"),
            (r"forallGroups", "P4 Γ global + output"),
        ],
        "operators": [
            # the σ and the probe share a line: findOne / contains inlined
            # under it is the probe, the rest of the line the σ and the
            # probe's compare
            (r"ht2\.findOne.*\[findOne\]$", "⋈ orders-lineitem: probe"),
            (r"l_shipdate\[i\] > c2", "σ l_shipdate (+ probe compare)"),
            (r"ht1\.contains.*\[contains\]$", "⋈ customer-orders: probe"),
            (r"o_orderdate\[i\] < c1", "σ o_orderdate (+ probe compare)"),
            (r"ht1\.hash|emplace_back\(ht1", "⋈ customer-orders: build (entries)"),
            (r"c_mktsegment\[i\] == c3", "σ c_mktsegment"),
            (r"parallel_insert\(entries1|ht1\.setSize", "⋈ customer-orders: build (insert)"),
            (r"emplace_back|make_tuple\(o_orderdate|ht2\.hash", "⋈ orders-lineitem: build (entries)"),
            (r"parallel_insert\(entries2|ht2\.setSize", "⋈ orders-lineitem: build (insert)"),
            (r"l_extendedprice\[i\] \* \(one", "π revenue"),
            (r"locals\.consume|preAggLocals", "Γ local"),
            (r"forallGroups", "Γ global + output"),
            SCAN_LOOP,
        ],
    },
    18: {
        "pipelines": [
            (r"^tbb::parallel_for\(tbb::blocked_range<size_t>\(0, li\.nrTuples", "P1 lineitem: subquery Γ local"),
            (r"^groupOp\.forallGroups", "P2 subquery Γ global, HAVING, ht1 entries"),
            (r"parallel_insert\(entries1|ht1\.setSize", "P2 ht1 insert"),
            (r"PARALLEL_SCAN\(cu\.nrTuples", "P3 customer: ht2 entries"),
            (r"parallel_insert\(entries2|ht2\.setSize", "P3 ht2 insert"),
            (r"PARALLEL_SELECT\(ord\.nrTuples", "P4 orders: semi ht1, probe ht2, ht3 entries"),
            (r"parallel_insert\(entries3|ht3\.setSize", "P4 ht3 insert"),
            (r"^tbb::parallel_for\($", "P5 lineitem: probe ht3, Γ local"),
            (r"finalGroupOp\.forallGroups", "P6 Γ global + output"),
        ],
        "operators": [
            (r"locals\.getGroup\(l_orderkey\[i\]\)|group \+= l_quantity", "Γ local (subquery or final)"),
            (r"getGroup\(tuple_cat", "Γ local (final)"),
            (r"^groupOp\.forallGroups|group\.v > threeHundret|for \(auto& group : block\)|entries1\.local",
             "subquery Γ global + HAVING"),
            (r"ht3\.findOne.*\[findOne\]$", "⋈ orders-lineitem: probe"),
            (r"ht3\.findOne", "⋈ orders-lineitem: probe (compare)"),
            (r"PARALLEL_SELECT.*\[contains\]$", "⋈ semi o_orderkey in subquery: probe"),
            (r"PARALLEL_SELECT.*\[findOne\]$", "⋈ customer-orders: probe"),
            (r"PARALLEL_SELECT", "orders: σ / ht3 entries"),
            (r"PARALLEL_SCAN", "⋈ customer-orders: build (entries)"),
            (r"parallel_insert\(entries1|ht1\.setSize", "⋈ semi: build (insert)"),
            (r"parallel_insert\(entries2|ht2\.setSize", "⋈ customer-orders: build (insert)"),
            (r"parallel_insert\(entries3|ht3\.setSize", "⋈ orders-lineitem: build (insert)"),
            (r"finalGroupOp\.forallGroups", "Γ global + output"),
            (r"^for \(size_t i = r\.begin\(\), end = r\.end\(\); i != end", "scan loop"),
            (r"^tbb::parallel_for", "scan loop"),
        ],
    },
}
