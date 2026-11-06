#pragma once
//===----------------------------------------------------------------------===//
// VW_GROUP_CONCAT_SIMD: HashGroup's compound key concatenation in one pass.
//
// HashGroup packs a row's key columns into one key of totalKeySize (K) bytes,
// column c at byte offset off_c (packedKeys, Concat_T: one strided memcpy
// pass per column). Here the packing is planned once per operator: for a
// block of R rows,
//   1. each column's R values are brought into one register, packed
//      (R * w_c bytes): a load for dense columns; for columns read through a
//      selection vector, scalar loads into a stack buffer (Stage) or vpgather
//      / NEON lane loads (Gather);
//   2. one byte permute per column, with a control built at plan time, moves
//      column c's bytes to r * K + off_c + j, and the columns are merged;
//   3. the R packed keys (R * K bytes) are stored once.
// The layout is byte-identical to Concat_T, so hashing, key equality and the
// spill are unchanged.
//
// Kernels: AVX-512 VBMI (vpermb over 64 bytes, R = 64 / K; Ice Lake and
// newer), AVX2 (vpshufb, two 16-byte lanes of R = 16 / K rows each), NEON
// (tbl, R = 16 / K). Without VBMI (Skylake-X, Cascade Lake) the AVX2 kernel
// is used. Columns of 1, 2, 4 or 8 bytes and keys up to one lane (16 bytes,
// 64 with VBMI); other shapes keep Concat_T (build() returns false).
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <cstdint>
#include "vectorwise/defs.hpp"

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__) &&   \
    defined(__AVX512VL__) && defined(__AVX512VBMI__)
#define VW_CONCAT_HAVE_VBMI 1
#endif
#if defined(__x86_64__) && defined(__AVX2__)
#define VW_CONCAT_HAVE_AVX2 1
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#define VW_CONCAT_HAVE_NEON 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_concat {

enum class Isa : uint8_t { Scalar, Avx2, Vbmi, Neon };
/// how a column read through a selection vector is brought into a register
enum class SelLoad : uint8_t {
   Stage, ///< scalar loads into a stack buffer, then a vector load
   Gather ///< vpgather (x86), lane loads (NEON)
};

constexpr size_t kMaxCols = 8;
/// bytes past n * K that run() may write (packedKeys is sized with them)
constexpr size_t kSlack = 64;

struct Plan {
   bool ok = false;
   Isa isa = Isa::Scalar;
   SelLoad selLoad = SelLoad::Stage;
   uint32_t keySize = 0;
   uint32_t cols = 0;
   /// rows per lane (AVX2: a block is two lanes)
   uint32_t rows = 0;
   /// rows a block reads from its first row on (loads past the block's own
   /// rows); blocks run while first row + lookahead <= n, the rest is scalar
   uint32_t lookahead = 0;
   uint8_t width[kMaxCols] = {};
   uint32_t offset[kMaxCols] = {};
   bool sel[kMaxCols] = {};
   /// per column: output byte b of a lane takes byte ctrl[c][b] of the
   /// column's register (AVX2 / NEON: 0x80 / 0xff = none of this column)
   alignas(64) uint8_t ctrl[kMaxCols][64] = {};
   /// VBMI: output bytes of a block that come from column c
   uint64_t lanes[kMaxCols] = {};
};

/// the widest kernel compiled in: VBMI, else AVX2, else NEON, else Scalar
Isa best();
bool available(Isa isa);
const char* name(Isa isa);

/// plan column c (width[c] bytes at offset[c] of a K-byte key, read through
/// a selection vector if sel[c]) for isa; false if the shape is not covered
/// (column width not 1/2/4/8, key wider than a lane, too many columns,
/// isa not compiled in)
bool build(Plan& p, Isa isa, SelLoad selLoad, size_t cols,
           const uint32_t* width, const uint32_t* offset, const bool* sel,
           uint32_t keySize);

/// pack rows [0, n): out[i * K + offset[c] ..] = column c of row i, where
/// row i of column c is data[c][i] or data[c][sel[c][i]]. Writes up to
/// kSlack bytes past n * K.
void run(const Plan& p, pos_t n, const char* const* data,
         const pos_t* const* sel, char* out);

/// the reference: one strided pass per column, as HashGroup::Concat_T
/// (width fixed per column for 1/2/4/8 bytes, memcpy of col width else)
void runColumns(size_t cols, const uint32_t* width, const uint32_t* offset,
                uint32_t keySize, pos_t n, const char* const* data,
                const pos_t* const* sel, char* out);

} // namespace simd_concat
} // namespace primitives
} // namespace vectorwise
