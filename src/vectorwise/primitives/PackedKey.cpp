// Packed-key primitives: concatenate two arbitrary-sized key columns into a
// single contiguous buffer, enabling a single hash/compare/scatter/gather
// per tuple instead of one per column.
//
// The pack primitive is templated on (SIZE_A, SIZE_B) so it works with any
// combination of key widths. The output buffer element size is SIZE_A+SIZE_B.

#include "common/runtime/Hash.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <cstring>
#include <functional>

using namespace types;

namespace vectorwise {
namespace primitives {

// ---------------------------------------------------------------------------
// Pack: two void* columns -> contiguous output buffer (via selection vector)
// Signature: F4 (n, sel, out, col_a, col_b)
// Template params SIZE_A, SIZE_B are the byte widths of each source column.
// Output element is SIZE_A + SIZE_B bytes.
// ---------------------------------------------------------------------------
template <size_t SIZE_A, size_t SIZE_B>
static pos_t pack_sel_(pos_t n, pos_t* RES sel,
                       void* RES out,
                       void* RES col_a,
                       void* RES col_b) {
   constexpr size_t OUT_SIZE = SIZE_A + SIZE_B;
   auto* dst = reinterpret_cast<uint8_t*>(out);
   auto* src_a = reinterpret_cast<uint8_t*>(col_a);
   auto* src_b = reinterpret_cast<uint8_t*>(col_b);
   for (uint64_t i = 0; i < n; ++i) {
      auto idx = sel[i];
      std::memcpy(dst + i * OUT_SIZE,            src_a + idx * SIZE_A, SIZE_A);
      std::memcpy(dst + i * OUT_SIZE + SIZE_A,   src_b + idx * SIZE_B, SIZE_B);
   }
   return n;
}

// Instantiate for Q1: Char<1> + Char<1> = 2 bytes
F4 pack_sel_void_1_1 = (F4)&pack_sel_<1, 1>;

// ---------------------------------------------------------------------------
// Hash: uint16_t packed key
// ---------------------------------------------------------------------------
#if HASH_SIZE == 32
#define DEFAULT_HASH runtime::MurMurHash3
#else
#define DEFAULT_HASH runtime::MurMurHash
#endif

F2 hash_uint16_t_col = (F2)&hash<uint16_t, DEFAULT_HASH>;
F3 hash_sel_uint16_t_col = (F3)&hash_sel<uint16_t, DEFAULT_HASH>;

// ---------------------------------------------------------------------------
// Key equality: uint16_t
// ---------------------------------------------------------------------------
NEQCheck keys_not_equal_uint16_t_col =
    (NEQCheck)&keys_not_equal<uint16_t>;
NEQCheckSel keys_not_equal_sel_uint16_t_col =
    (NEQCheckSel)&keys_not_equal_sel<uint16_t>;
NEQCheckRow keys_not_equal_row_uint16_t_col =
    (NEQCheckRow)&keys_not_equal_row<uint16_t>;

// ---------------------------------------------------------------------------
// Partition: uint16_t
// ---------------------------------------------------------------------------
FPartitionByKey partition_by_key_uint16_t_col =
    (FPartitionByKey)&partition_by_key<uint16_t>;
FPartitionByKeySel partition_by_key_sel_uint16_t_col =
    (FPartitionByKeySel)&partition_by_key_sel<uint16_t>;
FPartitionByKeyRow partition_by_key_row_uint16_t_col =
    (FPartitionByKeyRow)&partition_by_key_row<uint16_t>;

// ---------------------------------------------------------------------------
// Scatter: uint16_t
// ---------------------------------------------------------------------------
FScatter scatter_uint16_t_col = (FScatter)&scatter<uint16_t>;
FScatterSel scatter_sel_uint16_t_col = (FScatterSel)&scatter_sel<uint16_t>;
FScatterSelRow scatter_sel_row_uint16_t_col =
    (FScatterSelRow)&scatter_sel_row<uint16_t>;

// ---------------------------------------------------------------------------
// Gather: uint16_t
// ---------------------------------------------------------------------------
FGather gather_col_uint16_t_col = (FGather)&gather_col<uint16_t>;
FGatherVal gather_val_uint16_t_col = (FGatherVal)&gather_val<uint16_t>;

// ---------------------------------------------------------------------------
// Unpack: gather a uint16_t from the HT entry and split back into two
// Char<1> output buffers.
// ---------------------------------------------------------------------------
static pos_t unpack_q1key_returnflag_(pos_t n, uint16_t** RES input,
                                       size_t offset, size_t* struct_size,
                                       Char<1>* RES out) {
   auto current = addBytes(*input, offset);
   auto size = *struct_size;
   for (size_t i = 0; i < n; ++i) {
      std::memcpy(&out[i].value, reinterpret_cast<char*>(current), 1);
      current = addBytes(current, size);
   }
   return n;
}
FGatherVal unpack_q1key_returnflag = (FGatherVal)&unpack_q1key_returnflag_;

static pos_t unpack_q1key_linestatus_(pos_t n, uint16_t** RES input,
                                       size_t offset, size_t* struct_size,
                                       Char<1>* RES out) {
   auto current = addBytes(*input, offset);
   auto size = *struct_size;
   for (size_t i = 0; i < n; ++i) {
      std::memcpy(&out[i].value, reinterpret_cast<char*>(current) + 1, 1);
      current = addBytes(current, size);
   }
   return n;
}
FGatherVal unpack_q1key_linestatus = (FGatherVal)&unpack_q1key_linestatus_;

} // namespace primitives
} // namespace vectorwise
