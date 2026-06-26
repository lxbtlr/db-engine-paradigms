#include "benchmarks/Primitives.hpp"
#include "common/runtime/SIMD.hpp"
#include <string.h>

#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>

// --- Custom Polyfills ---
// Using custom prefixes to absolutely guarantee no redefinition 
// collisions with your specific 3rdparty simde version.

static inline simde__mmask8 custom_cmplt_epi64_mask(simde__m512i a, simde__m512i b) {
    alignas(64) int64_t arr_a[8], arr_b[8];
    simde_mm512_storeu_si512((simde__m512i*)arr_a, a);
    simde_mm512_storeu_si512((simde__m512i*)arr_b, b);
    simde__mmask8 mask = 0;
    for(int i=0; i<8; ++i) if (arr_a[i] < arr_b[i]) mask |= (1 << i);
    return mask;
}

static inline simde__mmask16 custom_cmplt_epi32_mask(simde__m512i a, simde__m512i b) {
    alignas(64) int32_t arr_a[16], arr_b[16];
    simde_mm512_storeu_si512((simde__m512i*)arr_a, a);
    simde_mm512_storeu_si512((simde__m512i*)arr_b, b);
    simde__mmask16 mask = 0;
    for(int i=0; i<16; ++i) if (arr_a[i] < arr_b[i]) mask |= (1 << i);
    return mask;
}

static inline simde__mmask32 custom_cmplt_epi16_mask(simde__m512i a, simde__m512i b) {
    alignas(64) int16_t arr_a[32], arr_b[32];
    simde_mm512_storeu_si512((simde__m512i*)arr_a, a);
    simde_mm512_storeu_si512((simde__m512i*)arr_b, b);
    simde__mmask32 mask = 0;
    for(int i=0; i<32; ++i) if (arr_a[i] < arr_b[i]) mask |= (1U << i);
    return mask;
}

static inline simde__m512i custom_i32gather_epi32(simde__m512i vindex, const void* base_addr, int scale) {
    alignas(64) int32_t idx[16];
    simde_mm512_storeu_si512((simde__m512i*)idx, vindex);
    alignas(64) int32_t val[16];
    for (int i = 0; i < 16; i++) {
        int32_t tmp = 0;
        memcpy(&tmp, (const char*)base_addr + idx[i] * scale, sizeof(int32_t));
        val[i] = tmp;
    }
    return simde_mm512_loadu_si512((simde__m512i const*)val);
}

// ------------------------------

using namespace std;
using vectorwise::pos_t;
using namespace vectorwise::primitives;

pos_t sel_less_int64_t_col_int64_t_val_avx512(pos_t n, pos_t* RES result,
                                                   int64_t* RES param1,
                                                   int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");
   uint64_t found = 0;
   size_t rest = n % 8;
   auto con = *param2;
   auto consts = simde_mm512_set1_epi64(con);
   auto ids = simde_mm256_set_epi32(7,6,5,4,3,2,1,0); 
   for (uint64_t i = 0; i < n - rest; i += 8) {
      auto in = Vec8u(param1);
      simde__mmask8 less = custom_cmplt_epi64_mask(in, consts);
      simde_mm256_mask_compressstoreu_epi32(result + found, less, ids);
      found += __builtin_popcount(less);
      ids = simde_mm256_add_epi32(ids, simde_mm256_set1_epi32(8));
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      if (param1[i] < con) result[found++] = i;
   }
   return found;
}


pos_t sel_less_int16_t_col_int16_t_val_avx512(pos_t n, pos_t* RES result,
                                                   int16_t* RES param1,
                                                   int16_t* RES param2) {
  static_assert(sizeof(pos_t) == 4,
                "This implementation only supports sizeof(pos_t) == 4");
  uint64_t found = 0;
  size_t rest = n % 32;
  auto con = *param2;
  auto consts = simde_mm512_set1_epi16(con);
  auto ids = simde_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  for (uint64_t i = 0; i < n - rest; i += 32) {
    auto in = Vec8u(param1);
    simde__mmask32 less = custom_cmplt_epi16_mask(in, consts);
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
    less >>= 16;
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    if (param1[i] < con) result[found++] = i;
  }
  return found;
}

pos_t sel_less_int8_t_col_int8_t_val_avx512(pos_t n, pos_t* RES result,
                                                   int8_t* RES param1,
                                                   int8_t* RES param2) {
  static_assert(sizeof(pos_t) == 4,
                "This implementation only supports sizeof(pos_t) == 4");
  uint64_t found = 0;
  size_t rest = n % 64;
  auto con = *param2;
  auto consts = simde_mm512_set1_epi16(con);
  auto ids = simde_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  for (uint64_t i = 0; i < n - rest; i += 64) {
    auto in = Vec8u(param1);
    // Your compiler output proved your SIMDe branch DOES have this specific mask natively, 
    // so we keep the native simde_ call for the 8-bit version.
    simde__mmask64 less = simde_mm512_cmplt_epi8_mask(in, consts);
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
    less >>= 16;
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
    less >>= 16;
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
    less >>= 16;
    simde_mm512_mask_compressstoreu_epi32(result + found, less, ids);
    found += __builtin_popcount(static_cast<simde__mmask16>(less));
    ids = simde_mm512_add_epi32(ids, simde_mm512_set1_epi32(16));
    less >>= 16;
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    if (param1[i] < con) result[found++] = i;
  }
  return found;
}

pos_t selsel_less_int32_t_col_int32_t_val_avx512(pos_t n, pos_t* RES inSel, pos_t* RES result, int32_t* RES param1,
                                                       int32_t* RES param2) {
  static_assert(sizeof(pos_t) == 4,"This implementation only supports sizeof(pos_t) == 4");

  uint64_t found = 0;
  size_t rest = n % 16;
  auto con = *param2;
  auto consts = simde_mm512_set1_epi32(con);
  for (uint64_t i = 0; i < n - rest; i += 16){
    Vec8u idxs(inSel + i);
    auto in = custom_i32gather_epi32(idxs, param1, 4);
    simde__mmask16 ge = custom_cmplt_epi32_mask(in, consts);
    simde_mm512_mask_compressstoreu_epi32(result+found, ge, idxs);
    found += __builtin_popcount(ge);
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    if (param1[idx] < con) result[found++] = idx;
  }
  return found;
}

pos_t selsel_less_int16_t_col_int16_t_val_avx512(pos_t n, pos_t* RES inSel, pos_t* RES result, int16_t* RES param1,
                                                 int16_t* RES param2) {
  static_assert(sizeof(pos_t) == 4,"This implementation only supports sizeof(pos_t) == 4");

  uint64_t found = 0;
  size_t rest = n % 16;
  auto con = *param2;
  auto consts = simde_mm512_set1_epi32(con);
  auto lower16 = simde_mm512_set1_epi32(0xffff);
  for (uint64_t i = 0; i < n - rest; i += 16){
    Vec8u idxs(inSel + i);
    auto in = custom_i32gather_epi32(idxs, param1, 2);
    in = simde_mm512_and_epi32(lower16, in);
    simde__mmask16 ge = custom_cmplt_epi32_mask(in, consts);
    simde_mm512_mask_compressstoreu_epi32(result+found, ge, idxs);
    found += __builtin_popcount(ge);
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    if (param1[idx] < con) result[found++] = idx;
  }
  return found;
}

pos_t selsel_less_int8_t_col_int8_t_val_avx512(pos_t n, pos_t* RES inSel, pos_t* RES result, int8_t* RES param1,
                                               int8_t* RES param2) {
  static_assert(sizeof(pos_t) == 4,"This implementation only supports sizeof(pos_t) == 4");

  uint64_t found = 0;
  size_t rest = n % 16;
  auto con = *param2;
  auto consts = simde_mm512_set1_epi32(con);
  auto lower8 = simde_mm512_set1_epi32(0xff);
  for (uint64_t i = 0; i < n - rest; i += 16){
    Vec8u idxs(inSel + i);
    auto in = custom_i32gather_epi32(idxs, param1, 1);
    in = simde_mm512_and_epi32(lower8, in);
    simde__mmask16 ge = custom_cmplt_epi32_mask(in, consts);
    simde_mm512_mask_compressstoreu_epi32(result+found, ge, idxs);
    found += __builtin_popcount(ge);
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    if (param1[idx] < con) result[found++] = idx;
  }
  return found;
}


void run(size_t n, size_t vecSize, PerfEvents& e) {
  using namespace vectorwise::primitives;

  auto* output = static_cast<pos_t*>(compat::aligned_alloc(64, vecSize*sizeof(pos_t)));
  vector<size_t> selectivities = {0,  1,  2,  5,  10, 20, 30, 40, 50,
                                  60, 70, 80, 90, 95, 98, 99, 100};

 auto r = [&](std::string name, auto cb, auto &val) {
   if (auto p = getenv("Sel")) {
     auto i = atoi(p);
     val = i;
     cout << "Selectivity: " << i << "\n";
     e.timeAndProfile(name, n, cb, 100);
   } else {
     for (auto i : selectivities) {
       val = i;
       cout << "Selectivity: " << i << "\n";
       e.timeAndProfile(name, n, cb, 100);
     }
   }
 };

 {
    int64_t val = 5;
    auto* input = static_cast<int64_t*>(compat::aligned_alloc(64, vecSize * sizeof(int64_t)));
    putRandom(input, vecSize, 0, 99);

    // 8 byte types
    cout << "typeSize: 8" << "\n";
    r("Scalar", [&]() { sel_less_int64_t_col_int64_t_val(n, output, input, &val); }, val);
    r("Predicated", [&]() { sel_less_int64_t_col_int64_t_val_bf(n, output, input, &val); }, val);
    r("SIMD", [&]() {sel_less_int64_t_col_int64_t_val_avx512(n, output, input, &val);}, val);
 }
 {
   int32_t val = 5;
   auto* input = static_cast<int32_t*>(compat::aligned_alloc(64, vecSize * sizeof(int32_t)));
   putRandom(input, vecSize, 0, 99);

   // 4 byte types
   cout << "typeSize: 4" << "\n";
   r("Scalar", [&]() { sel_less_int32_t_col_int32_t_val(n, output, input, &val); }, val);
   r("Predicated", [&]() { sel_less_int32_t_col_int32_t_val_bf(n, output, input, &val); }, val);
   r("SIMD", [&]() {sel_less_int32_t_col_int32_t_val_avx512(n, output, input, &val);}, val);
 }
 {
   int16_t val = 5;
   auto* input = static_cast<int16_t*>(compat::aligned_alloc(64, vecSize * sizeof(int16_t)));
   putRandom(input, vecSize, 0, 99);

   // 2 byte types
   cout << "typeSize: 2" << "\n";
   r("Scalar", [&]() { sel_less_int16_t_col_int16_t_val(n, output, input, &val); }, val);
   r("Predicated", [&]() { sel_less_int16_t_col_int16_t_val_bf(n, output, input, &val); }, val);
   r("SIMD", [&]() {sel_less_int16_t_col_int16_t_val_avx512(n, output, input, &val);}, val);
 }
 {
   int8_t val = 5;
   auto* input = static_cast<int8_t*>(compat::aligned_alloc(64, vecSize * sizeof(int8_t)));
   putRandom(input, vecSize, 0, 99);

   // 2 byte types
   cout << "typeSize: 1" << "\n";
   r("Scalar", [&]() { sel_less_int8_t_col_int8_t_val(n, output, input, &val); }, val);
   r("Predicated", [&]() { sel_less_int8_t_col_int8_t_val_bf(n, output, input, &val); }, val);
   r("SIMD", [&]() {sel_less_int8_t_col_int8_t_val_avx512(n, output, input, &val);}, val);
 }

 //    Followup selection
    {
      auto* input = static_cast<int64_t*>(compat::aligned_alloc(64, vecSize * sizeof(int64_t)));
      auto* sel = static_cast<pos_t*>(compat::aligned_alloc(64, n*sizeof(pos_t)));
      auto r = [&](std::string name, auto cb) {
        for (auto i : selectivities) {
          int64_t valInner = i;
          putRandom(input, vecSize, 0, 99);
          auto selected = sel_less_int64_t_col_int64_t_val(n, sel, input, &valInner);
          putRandom(selected, sel, input,  0, 99);
          cout << "Selectivity: " << i << "\n";
          e.timeAndProfile(name, selected, [cb, selected](){cb(selected);}, 100);
        }
      };

      int64_t val = 40;

      cout << "typeSize: 8" << "\n";
      r("Scalar", [&](pos_t n) { selsel_less_int64_t_col_int64_t_val(n, sel, output, input, &val); });
      r("Predicated", [&](pos_t n) { selsel_less_int64_t_col_int64_t_val_bf(n, sel, output, input, &val); });
      r("SIMD", [&](pos_t n) { selsel_less_int64_t_col_int64_t_val_avx512(n, sel, output, input, &val);});
    }
    {
      auto* input = static_cast<int32_t*>(compat::aligned_alloc(64, vecSize * sizeof(int32_t)));
      auto* sel = static_cast<pos_t*>(compat::aligned_alloc(64, n*sizeof(pos_t)));
      auto r = [&](std::string name, auto cb) {
        for (auto i : selectivities) {
          int32_t valInner = i;
          putRandom(input, vecSize, 0, 99);
          auto selected = sel_less_int32_t_col_int32_t_val(n, sel, input, &valInner);
          putRandom(selected, sel, input,  0, 99);
          cout << "Selectivity: " << i << "\n";
          e.timeAndProfile(name, selected, [cb, selected](){cb(selected);}, 100);
        }
      };


      int32_t val = 40;

      cout << "typeSize: 4" << "\n";
      r("Scalar", [&](pos_t n) { selsel_less_int32_t_col_int32_t_val(n, sel, output, input, &val); });
      r("Predicated", [&](pos_t n) { selsel_less_int32_t_col_int32_t_val_bf(n, sel, output, input, &val); });
      r("SIMD", [&](pos_t n) { selsel_less_int32_t_col_int32_t_val_avx512(n, sel, output, input, &val);});
    }
    {
      auto* input = static_cast<int16_t*>(compat::aligned_alloc(16, vecSize * sizeof(int16_t)));
      auto* sel = static_cast<pos_t*>(compat::aligned_alloc(16, n*sizeof(pos_t)));
      auto r = [&](std::string name, auto cb) {
        for (auto i : selectivities) {
          int16_t valInner = i;
          putRandom(input, vecSize, 0, 99);
          auto selected = sel_less_int16_t_col_int16_t_val(n, sel, input, &valInner);
          putRandom(selected, sel, input,  0, 99);
          cout << "Selectivity: " << i << "\n";
          e.timeAndProfile(name, selected, [cb, selected](){cb(selected);}, 100);
        }
      };

      int16_t val = 40;

      // 8 byte types
      cout << "typeSize: 2" << "\n";
      r("Scalar", [&](pos_t n) { selsel_less_int16_t_col_int16_t_val(n, sel, output, input, &val); });
      r("Predicated", [&](pos_t n) { selsel_less_int16_t_col_int16_t_val_bf(n, sel, output, input, &val); });
      r("SIMD", [&](pos_t n) { selsel_less_int16_t_col_int16_t_val_avx512(n, sel, output, input, &val);});
    }
    {
      auto* input = static_cast<int8_t*>(compat::aligned_alloc(8, vecSize * sizeof(int8_t)));
      auto* sel = static_cast<pos_t*>(compat::aligned_alloc(8, n*sizeof(pos_t)));
      auto r = [&](std::string name, auto cb) {
        for (auto i : selectivities) {
          int8_t valInner = i;
          putRandom(input, vecSize, 0, 99);
          auto selected = sel_less_int8_t_col_int8_t_val(n, sel, input, &valInner);
          putRandom(selected, sel, input,  0, 99);
          cout << "Selectivity: " << i << "\n";
          e.timeAndProfile(name, selected, [cb, selected](){cb(selected);}, 100);
        }
      };

      int8_t val = 40;

      // 8 byte types
      cout << "typeSize: 1" << "\n";
      r("Scalar", [&](pos_t n) { selsel_less_int8_t_col_int8_t_val(n, sel, output, input, &val); });
      r("Predicated", [&](pos_t n) { selsel_less_int8_t_col_int8_t_val_bf(n, sel, output, input, &val); });
      r("SIMD", [&](pos_t n) { selsel_less_int8_t_col_int8_t_val_avx512(n, sel, output, input, &val);});
    }
}

int main() {
   // size_t vecSize = 1024 * 1024; // 1G variant
   size_t vecSize = 1024 * 8;
   size_t n = vecSize;
   PerfEvents e;
   run(n, vecSize, e);
   return 0;
}
