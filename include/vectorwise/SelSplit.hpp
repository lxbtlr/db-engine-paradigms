#pragma once
// VW_SEL_SPLIT: the branch-free selection templates as two passes, so the
// compiler vectorizes the comparison (autovec addendum A4). The one-pass
// loop stores the candidate position and advances the output cursor by the
// comparison, a compress neither gcc nor clang vectorizes. Here, per chunk:
//   mask pass:     mask[i] = decision(i), one byte per row (fixed-length
//                  output: vectorizes)
//   compress pass: 8 mask bytes at a time to an 8-bit mask (one multiply),
//                  the set bits' positions from a 256 x 8 table, all 8
//                  positions stored, the cursor advanced by popcount
// Only for arithmetic and Date columns; Char/Varchar compares do not
// vectorize and keep the one-pass loop. The SIMD selection kernels still win
// wherever their options pick them.
#include "common/runtime/Types.hpp"
#include "vectorwise/defs.hpp"
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace vectorwise {
namespace primitives {
namespace sel_split {

template <typename T>
constexpr bool kOn =
#ifdef VW_SEL_SPLIT
    std::is_arithmetic<T>::value || std::is_same<T, types::Date>::value;
#else
    false;
#endif

/// rows per mask pass: the mask stays in L1
constexpr pos_t kChunk = 1024;

/// pos[m][j]: position of the j-th set bit of m (0 past the last one)
struct Lut {
   uint8_t pos[256][8];
   constexpr Lut() : pos() {
      for (unsigned m = 0; m < 256; m++) {
         unsigned k = 0;
         for (unsigned b = 0; b < 8; b++)
            if (m & (1u << b)) pos[m][k++] = uint8_t(b);
      }
   }
};
inline constexpr Lut lut{};

/// positions with a set mask byte (mask bytes are 0 or 1): base + i, or
/// inSel[base + i] with SelIn. Writes at most at out[i + 7] for a full block
/// of 8 at row i, so out needs no slack beyond n; it may alias inSel, as
/// every block reads its 8 entries before it writes.
template <bool SelIn>
inline pos_t compress(pos_t n, const uint8_t* mask, pos_t base,
                      const pos_t* inSel, pos_t* out) {
   pos_t found = 0, i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64_t x;
      std::memcpy(&x, mask + i, 8);
      // byte j (0 or 1) to bit j: the partial products do not overlap
      const unsigned m = unsigned((x * 0x0102040810204080ull) >> 56);
      const uint8_t* p = lut.pos[m];
      // one contiguous 8-entry block: the stores and the widening add
      // vectorize (SLP)
      const size_t row = size_t(base) + i;
      pos_t v[8];
      for (int j = 0; j < 8; j++)
         v[j] = SelIn ? inSel[row + p[j]] : pos_t(row + p[j]);
      pos_t* o = out + found;
      for (int j = 0; j < 8; j++) o[j] = v[j];
      found += __builtin_popcount(m);
   }
   for (; i < n; i++) {
      out[found] = SelIn ? inSel[base + i] : pos_t(base + i);
      found += mask[i];
   }
   return found;
}

/// the split selection: decide(idx) for idx = i or inSel[i]; idx is a
/// size_t so the mask loop's index cannot wrap, which keeps it vectorizable
template <bool SelIn, typename Decide>
inline pos_t select(pos_t n, const pos_t* inSel, pos_t* out, Decide decide) {
   uint8_t mask[kChunk];
   pos_t found = 0;
   for (size_t b = 0; b < n; b += kChunk) { // size_t: no wrap with VW_POS_16
      const pos_t m = pos_t(n - b < kChunk ? n - b : kChunk);
      for (size_t i = 0; i < m; i++)
         mask[i] = decide(SelIn ? size_t(inSel[b + i]) : b + i);
      found += compress<SelIn>(m, mask, pos_t(b), inSel, out + found);
   }
   return found;
}

} // namespace sel_split
} // namespace primitives
} // namespace vectorwise
