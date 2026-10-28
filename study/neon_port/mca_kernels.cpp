// llvm-mca harness: one extern "C" noinline wrapper per kernel whose hot loop
// study/neon_port/mca.sh extracts and runs through llvm-mca. The scalar
// wrappers are the engine templates or the engine loops run_neonbench copies
// (cited there); the neon/sve ones call include/vectorwise/SimdNeon.hpp.
// Compile for aarch64 only (mca.sh does).
#include "common/runtime/Hash.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdNeon.hpp"

using namespace vectorwise;
using namespace vectorwise::primitives;
#define W extern "C" __attribute__((noinline, flatten))

// sel (VW_SIMD_SEL)
W pos_t sel_i32_scalar_bf(pos_t n, pos_t* r, int32_t* a, int32_t* c) { return sel_col_val_bf<int32_t, std::less>(n, r, a, c); }
W pos_t sel_i32_neon4(pos_t n, pos_t* r, int32_t* a, int32_t* c) { return neon::sel_col_val<int32_t, std::less>(n, r, a, c); }
W pos_t sel_i32_neon16(pos_t n, pos_t* r, int32_t* a, int32_t* c) { return neon::sel_col_val_x16<int32_t, std::less>(n, r, a, c); }
W pos_t sel_i64_scalar_bf(pos_t n, pos_t* r, int64_t* a, int64_t* c) { return sel_col_val_bf<int64_t, std::less>(n, r, a, c); }
W pos_t sel_i64_neon4(pos_t n, pos_t* r, int64_t* a, int64_t* c) { return neon::sel_col_val<int64_t, std::less>(n, r, a, c); }
W pos_t sel_i64_neon16(pos_t n, pos_t* r, int64_t* a, int64_t* c) { return neon::sel_col_val_x16<int64_t, std::less>(n, r, a, c); }
// selsel
W pos_t selsel_i32_scalar_bf(pos_t n, pos_t* s, pos_t* r, int32_t* a, int32_t* c) { return selsel_col_val_bf<int32_t, std::less>(n, s, r, a, c); }
W pos_t selsel_i32_neon(pos_t n, pos_t* s, pos_t* r, int32_t* a, int32_t* c) { return neon::selsel_col_val<int32_t, std::less>(n, s, r, a, c); }
// hash (VW_SIMD_HASH; CRC for scale)
W void hash_murmur_scalar(pos_t n, uint64_t* o, int32_t* k) { primitives::hash<int32_t, runtime::MurMurHash>(n, o, k); }
W void hash_murmur_neon(pos_t n, uint64_t* o, int32_t* k) { neon::murmur_hash_i32(n, o, k, primitives::seed); }
W void hash_crc_scalar(pos_t n, uint64_t* o, int32_t* k) { primitives::hash<int32_t, runtime::CRC32Hash>(n, o, k); }
// proj (VW_PROJ_DENSE)
W void proj_mul_scalar(pos_t m, int64_t* o, int64_t* a, int64_t* b) { for (pos_t i = 0; i < m; ++i) o[i] = a[i] * b[i]; }
W void proj_mul_neon(pos_t n, pos_t* s, int64_t* o, int64_t* a, int64_t* b) { neon::proj_dense_multiplies_col_col(n, s, o, a, b); }
W void proj_minus_scalar(pos_t m, int64_t* o, int64_t c, int64_t* b) { for (pos_t i = 0; i < m; ++i) o[i] = c - b[i]; }
W void proj_minus_neon(pos_t n, pos_t* s, int64_t* o, int64_t* c, int64_t* b) { neon::proj_dense_minus_val_col(n, s, o, c, b); }
W void proj_gather_minus_scalar(pos_t n, pos_t* s, int64_t* o, int64_t* c, int64_t* b) { proj_sel_val_col<int64_t, std::minus>(n, s, o, c, b); }
// runheads (VW_GROUP_RUN_HEADS pass 1)
W pos_t runheads_u32_scalar(pos_t n, const uint32_t* RES k, pos_t* RES hd, uint8_t* RES fl) {
   pos_t cnt = 1;
   for (pos_t i = 1; i < n; ++i) {
      const bool head = k[i] != k[i - 1];
      fl[i] = head;
      hd[cnt] = i;
      cnt += head;
   }
   return cnt;
}
W pos_t runheads_u32_neon(pos_t n, const uint32_t* k, pos_t* hd, uint8_t* fl) { return neon::run_heads<uint32_t>(n, k, hd, fl, 1, 1); }
W pos_t runheads_u16_neon(pos_t n, const uint16_t* k, pos_t* hd, uint8_t* fl) { return neon::run_heads<uint16_t>(n, k, hd, fl, 1, 1); }
// bloom (VW_JOIN_BLOOM)
static inline uint64_t bloomBits(uint64_t h) {
   return (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
          (uint64_t(1) << ((h >> 12) & 63)) | (uint64_t(1) << ((h >> 18) & 63));
}
W size_t bloom_scalar_bf(size_t n, const uint64_t* H, const uint64_t* B, uint64_t mask, pos_t* S, uint64_t* HS) {
   size_t m = 0;
   for (size_t i = 0; i < n; ++i) {
      const auto h = H[i];
      const uint64_t b = bloomBits(h);
      S[m] = pos_t(i), HS[m] = h;
      m += (B[(h >> 40) & mask] & b) == b;
   }
   return m;
}
W size_t bloom_neon(size_t n, const uint64_t* H, const uint64_t* B, uint64_t mask, pos_t* S, uint64_t* HS) { return neon::bloom_filter(n, H, B, mask, S, HS); }
// semi (VW_JOIN_SEMI)
W size_t semi_scalar_bf(size_t n, const int32_t* k, const uint32_t* Bt, int32_t lo, uint32_t span, pos_t* o) {
   size_t found = 0;
   for (size_t i = 0; i < n; ++i) {
      const uint32_t d = uint32_t(k[i]) - uint32_t(lo);
      const uint32_t dc = d <= span ? d : span;
      o[found] = pos_t(i);
      found += (d <= span) & (Bt[dc >> 5] >> (dc & 31));
   }
   return found;
}
W size_t semi_neon(size_t n, const int32_t* k, const uint32_t* Bt, int32_t lo, uint32_t span, pos_t* o) { return neon::semi_probe(n, k, Bt, lo, span, o); }
#ifdef VW_HAVE_SVE
W pos_t sel_i32_sve(pos_t n, pos_t* r, const int32_t* a, int32_t c) { return neon::sve_sel_col_val_i32<std::less>(n, r, a, c); }
W pos_t selsel_i32_sve(pos_t n, const pos_t* s, pos_t* r, const int32_t* a, int32_t c) { return neon::sve_selsel_col_val_i32<std::less>(n, s, r, a, c); }
W void hash_murmur_sve(size_t n, uint64_t* o, const int32_t* k) { neon::sve_murmur_hash_i32(n, o, k, primitives::seed); }
W pos_t proj_mul_sve(pos_t n, pos_t* s, int64_t* o, int64_t* a, int64_t* b) { return neon::sve_proj_dense_multiplies(n, s, o, a, b); }
#endif
// compile check of the wiring skeleton's picker (VW_SIMD_SEL off: scalar)
W void* pick_sel_check() {
   return (void*)neon::pick_sel_col_val<int32_t, std::less>(&sel_col_val_bf<int32_t, std::less>);
}
