-- SSB reference answers for src/test/ssb.cpp and src/test/q*_ssb_expected.csv.
-- Run through scripts/ssb_expected.sh, which substitutes $D (data directory
-- with the five .tbl files) and $OUT (output directory) and strips the \r
-- that DuckDB's CSV mode writes.
create view lineorder as select * from read_csv('$D/lineorder.tbl', delim='|', header=false, quote='', names=['lo_orderkey','lo_linenumber','lo_custkey','lo_partkey','lo_suppkey','lo_orderdate','lo_orderpriority','lo_shippriority','lo_quantity','lo_extendedprice','lo_ordtotalprice','lo_discount','lo_revenue','lo_supplycost','lo_tax','lo_commitdate','lo_shopmode','x']);
create view part as select * from read_csv('$D/part.tbl', delim='|', header=false, quote='', names=['p_partkey','p_name','p_mfgr','p_category','p_brand1','p_color','p_type','p_size','p_container','x']);
create view supplier as select * from read_csv('$D/supplier.tbl', delim='|', header=false, quote='', names=['s_suppkey','s_name','s_address','s_city','s_nation','s_region','s_phone','x']);
create view customer as select * from read_csv('$D/customer.tbl', delim='|', header=false, quote='', names=['c_custkey','c_name','c_address','c_city','c_nation','c_region','c_phone','c_mktsegment','x']);
create view date as select * from read_csv('$D/date.tbl', delim='|', header=false, quote='', names=['d_datekey','d_date','d_dayofweek','d_month','d_year','d_yearmonthnum','d_yearmonth','d_daynuminweek','d_daynuminmonth','d_daynuminyear','d_monthnuminyear','d_weeknuminyear','d_sellingseason','d_lastdayinweekfl','d_lastdayinmonthfl','d_holidayfl','d_weekdayfl','x']);
.mode csv
.headers off
.output $OUT/q1x.txt
select 'q11', sum(lo_extendedprice*lo_discount) from lineorder, date where lo_orderdate=d_datekey and d_year=1993 and lo_discount between 1 and 3 and lo_quantity < 25
union all select 'q12', sum(lo_extendedprice*lo_discount) from lineorder, date where lo_orderdate=d_datekey and d_yearmonthnum=199401 and lo_discount between 4 and 6 and lo_quantity between 26 and 35
union all select 'q13', sum(lo_extendedprice*lo_discount) from lineorder, date where lo_orderdate=d_datekey and d_weeknuminyear=6 and d_year=1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35;
.output $OUT/q21_ssb_expected.csv
select d_year, p_brand1, printf('%d.00', sum(lo_revenue)) from lineorder, date, part, supplier where lo_orderdate=d_datekey and lo_partkey=p_partkey and lo_suppkey=s_suppkey and p_category='MFGR#12' and s_region='AMERICA' group by d_year, p_brand1;
.output $OUT/q22_ssb_expected.csv
select d_year, p_brand1, printf('%d.00', sum(lo_revenue)) from lineorder, date, part, supplier where lo_orderdate=d_datekey and lo_partkey=p_partkey and lo_suppkey=s_suppkey and p_brand1 between 'MFGR#2221' and 'MFGR#2228' and s_region='ASIA' group by d_year, p_brand1;
.output $OUT/q23_ssb_expected.csv
select d_year, p_brand1, printf('%d.00', sum(lo_revenue)) from lineorder, date, part, supplier where lo_orderdate=d_datekey and lo_partkey=p_partkey and lo_suppkey=s_suppkey and p_brand1='MFGR#2239' and s_region='EUROPE' group by d_year, p_brand1;
.output $OUT/q31_ssb_expected.csv
select c_nation, s_nation, d_year, printf('%d.00', sum(lo_revenue)) from lineorder, customer, supplier, date where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_orderdate=d_datekey and c_region='ASIA' and s_region='ASIA' and d_year between 1992 and 1997 group by c_nation, s_nation, d_year;
.output $OUT/q32_ssb_expected.csv
select c_city, s_city, d_year, printf('%d.00', sum(lo_revenue)) from lineorder, customer, supplier, date where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_orderdate=d_datekey and c_nation='UNITED STATES' and s_nation='UNITED STATES' and d_year between 1992 and 1997 group by c_city, s_city, d_year;
.output $OUT/q33_ssb_expected.csv
select c_city, s_city, d_year, printf('%d.00', sum(lo_revenue)) from lineorder, customer, supplier, date where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_orderdate=d_datekey and c_city in ('UNITED KI1','UNITED KI5') and s_city in ('UNITED KI1','UNITED KI5') and d_year between 1992 and 1997 group by c_city, s_city, d_year;
.output $OUT/q34_ssb_expected.csv
select c_city, s_city, d_year, printf('%d.00', sum(lo_revenue)) from lineorder, customer, supplier, date where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_orderdate=d_datekey and c_city in ('UNITED KI1','UNITED KI5') and s_city in ('UNITED KI1','UNITED KI5') and d_yearmonth='Dec1997' group by c_city, s_city, d_year;
.output $OUT/q41_ssb_expected.csv
select d_year, c_nation, printf('%d.00', sum(lo_revenue - lo_supplycost)) from date, customer, supplier, part, lineorder where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_partkey=p_partkey and lo_orderdate=d_datekey and c_region='AMERICA' and s_region='AMERICA' and p_mfgr in ('MFGR#1','MFGR#2') group by d_year, c_nation;
.output $OUT/q42_ssb_expected.csv
select d_year, s_nation, p_category, printf('%d.00', sum(lo_revenue - lo_supplycost)) from date, customer, supplier, part, lineorder where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_partkey=p_partkey and lo_orderdate=d_datekey and c_region='AMERICA' and s_region='AMERICA' and d_year in (1997,1998) and p_mfgr in ('MFGR#1','MFGR#2') group by d_year, s_nation, p_category;
.output $OUT/q43_ssb_expected.csv
select d_year, s_city, p_brand1, printf('%d.00', sum(lo_revenue - lo_supplycost)) from date, customer, supplier, part, lineorder where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_partkey=p_partkey and lo_orderdate=d_datekey and s_nation='UNITED STATES' and d_year in (1997,1998) and p_category='MFGR#14' and c_region='AMERICA' group by d_year, s_city, p_brand1;
