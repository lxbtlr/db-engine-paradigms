#pragma once

#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Database.hpp"

// Use CFG_NUM_REGIONS from CMake; fallback matches Concurrency.hpp default
#ifndef CFG_NUM_REGIONS
#ifndef CFG_SOCKETS_COUNT
#define CFG_SOCKETS_COUNT 4
#endif
#define CFG_NUM_REGIONS CFG_SOCKETS_COUNT
#endif

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
/// nActiveRegions: only populate shards 0..nActiveRegions-1 (data placed only
/// on nodes that have threads).  Defaults to NUM_NUMA_REGIONS.
void numaShardRelation(Relation& rel, size_t nActiveRegions = CFG_NUM_REGIONS);

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
