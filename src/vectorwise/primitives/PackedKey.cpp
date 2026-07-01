// Packed-key primitives for Q1's (l_returnflag, l_linestatus) composite key.
// Packs two Char<1> columns into a single uint16_t, enabling a single hash,
// compare, partition, scatter, and gather call instead of one per column.

#include "common/runtime/Hash.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <cstring>
#include <functional>

using namespace types;

namespace vectorwise {
namespace primitives {

// ---------------------------------------------------------------------------
// Pack: two Char<1> columns -> uint16_t buffer (via selection vector)
// Signature: F4 (n, sel, outKeys, returnflag_col, linestatus_col)
// ---------------------------------------------------------------------------
static pos_t pack_sel_q1keys_(pos_t n, pos_t* RES sel,
                               uint16_t* RES out,
                               Char<1>* RES col_a,
                               Char<1>* RES col_b) {
   for (uint64_t i = 0; i < n; ++i) {
      auto idx = sel[i];
      uint16_t k;
      std::memcpy(reinterpret_cast<char*>(&k),     &col_a[idx].value, 1);
      std::memcpy(reinterpret_cast<char*>(&k) + 1, &col_b[idx].value, 1);
      out[i] = k;
   }
   return n;
}
F4 pack_sel_q1keys = (F4)&pack_sel_q1keys_;

// ---------------------------------------------------------------------------
// Hash: uint16_t packed key -> groupHashes (with selection vector)
// Signature: F3 (n, sel, groupHashes, packedKeys)
// Uses hash_sel template, but since we produce a dense packed key buffer
// (not column-indexed), we use a direct hash without selection.
// Actually: groupHash is wired as F3(n, sel, groupHashes, col).
// For the packed key, col IS the dense buffer (indexed 0..n-1), but sel
// is needed because the hash_sel template does result[i] = hash(input[sel[i]]).
// We want result[i] = hash(input[i]) since pack already applied sel.
// So we use the non-sel F2 variant instead via the no-sel addKey overload,
// OR we use hash_sel with an identity selection. Simpler: just use a direct
// hash primitive.
// ---------------------------------------------------------------------------

// F2: hash_uint16_t_col(n, groupHashes, packedKeys)
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
// Char<1> output buffers. This replaces two separate gather_val_Char_1_col
// calls with a single pass.
//
// Signature matches FGatherVal:
//   (pos_t n, void** input, size_t offset, size_t* struct_size, void* out)
// But we need TWO outputs. We'll call this twice with different output
// buffers, using two different unpack primitives (high byte, low byte).
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
