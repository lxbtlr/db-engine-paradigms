#pragma once
#ifdef NUMA_ALLOC

#include "common/runtime/Database.hpp"

namespace runtime {

/// Replicate all columns of a relation per NUMA region.
/// Each replica is allocated on its region via first-touch (thread pinned
/// to a core on that region does mmap + memcpy).
void numaReplicateRelation(Relation& rel);

/// Free all NUMA replicas for a relation.
void numaFreeReplicas(Relation& rel);

} // namespace runtime

#endif // NUMA_ALLOC
