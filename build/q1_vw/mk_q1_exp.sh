#!/bin/bash

pushd /home/alexb/swole/db-engines/build/q1_vw
#pushd /home/alexb/swole/db-engines/build/new_aggr

#  - Baseline (original fused HashGroup::next()):
#  cmake -DCMAKE_BUILD_TYPE=Release ../..
#- Call q1_vectorwise().
  


rm -rf CMake*

cmake -DCMAKE_BUILD_TYPE=Release -DNUMA_LATENCY=ON ../..

make -j22 run_tpch 2> /dev/null

mv run_tpch control_run_tpch


#  - Option 2 (split operators, op-outer/tuple-inner — the existing primitive loop behavior):
#  cmake -DCMAKE_BUILD_TYPE=Release -DVW_SPLIT_HASHGROUP=ON ../..
#  - Call q1_vectorwise_split().
  

rm -rf CMake*

cmake -DCMAKE_BUILD_TYPE=Release -DVW_SPLIT_HASHGROUP=ON ../..

make -j22 run_tpch 2> /dev/null

mv run_tpch option2_run_tpch


#  - Option 1 (split operators, tuple-outer/op-inner):
#  cmake -DCMAKE_BUILD_TYPE=Release -DVW_SPLIT_HASHGROUP=ON -DVW_AGGR_TUPLE_OUTER=ON ../..
#  - Call q1_vectorwise_split().

rm -rf CMake*

cmake -DCMAKE_BUILD_TYPE=Release -DVW_SPLIT_HASHGROUP=ON -DVW_AGGR_TUPLE_OUTER=ON ../..

make -j22 run_tpch 

mv run_tpch option1_run_tpch

popd

