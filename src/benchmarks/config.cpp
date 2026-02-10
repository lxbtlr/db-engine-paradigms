#include "benchmarks/tpch/Queries.hpp"
#include "vectorwise/Primitives.hpp"


ExperimentConfig conf;

vectorwise::primitives::F2 ExperimentConfig::hash_int32_t_col() {


#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
   if (useSimdHash) return vectorwise::primitives::hash4_int32_t_col;
#endif
   return vectorwise::primitives::hash_int32_t_col;
}
vectorwise::primitives::F3 ExperimentConfig::hash_sel_int32_t_col() {


#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
   if (useSimdHash) return vectorwise::primitives::hash4_sel_int32_t_col;
#endif
   return vectorwise::primitives::hash_sel_int32_t_col;
}
vectorwise::primitives::F2 ExperimentConfig::rehash_int32_t_col() {


#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
   if (useSimdHash) return vectorwise::primitives::rehash4_int32_t_col;
#endif
   return vectorwise::primitives::rehash_int32_t_col;
}
vectorwise::primitives::F3 ExperimentConfig::rehash_sel_int32_t_col() {

#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)

   if (useSimdHash) return vectorwise::primitives::rehash4_sel_int32_t_col;
#endif
   return vectorwise::primitives::rehash_sel_int32_t_col;
}
vectorwise::primitives::F4 ExperimentConfig::proj_sel_minus_int64_t_val_int64_t_col(){


#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdProj)
  return vectorwise::primitives::proj_sel8_minus_int64_t_val_int64_t_col;
#endif
  return vectorwise::primitives::proj_sel_minus_int64_t_val_int64_t_col;
}
vectorwise::primitives::F4 ExperimentConfig::proj_sel_plus_int64_t_col_int64_t_val(){
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdProj)
    return vectorwise::primitives::proj_sel8_plus_int64_t_col_int64_t_val;
#endif
  return vectorwise::primitives::proj_sel_plus_int64_t_col_int64_t_val;
}
vectorwise::primitives::F3 ExperimentConfig::proj_multiplies_int64_t_col_int64_t_col(){
#if defined(__AVX512VL__) || defined(SIMDE_X86_AVX512VL_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdProj)
  return vectorwise::primitives::proj8_multiplies_int64_t_col_int64_t_col;
#endif
  return vectorwise::primitives::proj_multiplies_int64_t_col_int64_t_col;
}
vectorwise::primitives::F4 ExperimentConfig::proj_multiplies_sel_int64_t_col_int64_t_col(){
#if defined(__AVX512VL__) || defined(SIMDE_X86_AVX512VL_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdProj)
  return vectorwise::primitives::proj8_multiplies_sel_int64_t_col_int64_t_col;
#endif
  return vectorwise::primitives::proj_multiplies_sel_int64_t_col_int64_t_col;
}
vectorwise::primitives::F3 ExperimentConfig::sel_less_int32_t_col_int32_t_val(){
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if(useSimdSel) return vectorwise::primitives::sel_less_int32_t_col_int32_t_val_avx512;
#endif
  return BF(vectorwise::primitives::sel_less_int32_t_col_int32_t_val);
}
vectorwise::primitives::F4 ExperimentConfig::selsel_greater_equal_int32_t_col_int32_t_val() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if(useSimdSel) return vectorwise::primitives::selsel_greater_equal_int32_t_col_int32_t_val_avx512;
#endif
  return BF(vectorwise::primitives::selsel_greater_equal_int32_t_col_int32_t_val);
}
vectorwise::primitives::F4 ExperimentConfig::selsel_less_int64_t_col_int64_t_val() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if(useSimdSel) return vectorwise::primitives::selsel_less_int64_t_col_int64_t_val_avx512;
#endif
  return BF(vectorwise::primitives::selsel_less_int64_t_col_int64_t_val);
}
vectorwise::primitives::F4 ExperimentConfig::selsel_greater_equal_int64_t_col_int64_t_val() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if(useSimdSel) return vectorwise::primitives::selsel_greater_equal_int64_t_col_int64_t_val_avx512;
#endif
  return BF(vectorwise::primitives::selsel_greater_equal_int64_t_col_int64_t_val);
}
vectorwise::primitives::F4 ExperimentConfig::selsel_less_equal_int64_t_col_int64_t_val() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if(useSimdSel) return vectorwise::primitives::selsel_less_equal_int64_t_col_int64_t_val_avx512;
#endif
  return BF(vectorwise::primitives::selsel_less_equal_int64_t_col_int64_t_val);
}

ExperimentConfig::joinFun ExperimentConfig::joinAll() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdJoin) return &vectorwise::Hashjoin::joinAllSIMD;
#endif
  char* v;
  if ((v = std::getenv("JoinBoncz")) && atoi(v) != 0)
    return &vectorwise::Hashjoin::joinBoncz;
  return &vectorwise::Hashjoin::joinAllParallel;
}

ExperimentConfig::joinFun ExperimentConfig::joinSel() {
#if defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  if (useSimdJoin) return &vectorwise::Hashjoin::joinSelSIMD;
#endif
  return &vectorwise::Hashjoin::joinSelParallel;
}
