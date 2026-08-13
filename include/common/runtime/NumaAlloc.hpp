#pragma once

#include "common/runtime/Database.hpp"

namespace runtime {

#ifdef NUMA_ALLOC
/// Replicate all columns of a relation per NUMA region.
/// Each replica is allocated on its region via first-touch (thread pinned
/// to a core on that region does mmap + memcpy).
void numaReplicateRelation(Relation& rel);

/// Free all NUMA replicas for a relation.
void numaFreeReplicas(Relation& rel);
#endif

#ifdef NUMA_SHARD
/// Shard all columns of a relation across NUMA regions.
/// Each region owns tuples [tupleBegin, tupleEnd) in an mbind'd anonymous
/// mapping.  Loader threads pinned to each region memcpy from the original
/// file-backed mmap.
void numaShardRelation(Relation& rel);

/// Free all NUMA shards for a relation.
void numaFreeShards(Relation& rel);
#endif

#if defined(NUMA_DEBUG) && (defined(NUMA_ALLOC) || defined(NUMA_SHARD))
/// Verify that replicated/sharded pages landed on the expected NUMA nodes.
/// Samples across each region's full extent at 2MB stride.
/// Aborts if any region has <95% of pages on its home node.
void verifyNumaPlacement(Relation& rel);
#endif

} // namespace runtime
