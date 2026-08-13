#pragma once
#include <stdexcept>
#include <sys/mman.h>

#ifdef NUMA_SHARD
#include <cerrno>
#include <cstring>
#include <numaif.h>
#endif

namespace mem {
inline void* malloc_huge(size_t size) {
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef __linux__
   madvise(p, size); //MADV_HUGEPAGE);
#endif
   // memset(p, 0, size);
   return p;
}

inline void free_huge(void* p, size_t size) {
   auto r = munmap(p, size);
   if (r) throw std::runtime_error("Memory unmapping failed.");
}

#ifdef NUMA_SHARD
/// Allocate anonymous memory bound to a specific NUMA node.
/// Tries MAP_HUGETLB first; falls back to base pages if unavailable.
/// Does NOT use MADV_HUGEPAGE (avoids THP defrag overhead).
inline void* malloc_numa(size_t size, int node) {
   // Round up to page boundary (4KB for base, 2MB for huge)
   constexpr size_t HUGEPAGE = 2 * 1024 * 1024;
   size_t allocSize = (size + HUGEPAGE - 1) & ~(HUGEPAGE - 1);

   // Try hugetlb first (same allocSize either way for consistent cleanup)
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
   if (p == MAP_FAILED) {
      // Fall back to base pages, keep hugepage-rounded size
      p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED)
         throw std::runtime_error("malloc_numa: mmap failed: " +
                                  std::string(std::strerror(errno)));
   }

   // Bind to the target node before any page is touched
   unsigned long nodemask = 1UL << node;
   int rc = mbind(p, allocSize, MPOL_BIND, &nodemask,
                  sizeof(nodemask) * 8, MPOL_MF_STRICT);
   if (rc != 0)
      throw std::runtime_error("malloc_numa: mbind to node " +
                               std::to_string(node) + " failed: " +
                               std::string(std::strerror(errno)));
   return p;
}

/// Return the actual mmap size used by malloc_numa for cleanup.
/// Must match the rounding logic in malloc_numa.
inline size_t malloc_numa_size(size_t size) {
   constexpr size_t HUGEPAGE = 2 * 1024 * 1024;
   size_t hugeSize = (size + HUGEPAGE - 1) & ~(HUGEPAGE - 1);
   // We can't know here whether hugetlb succeeded. For munmap, passing the
   // larger (hugepage-rounded) size is safe even for base-page mappings as
   // long as the mmap was that size. So malloc_numa always uses hugepage
   // rounding for the mmap size, even on fallback.
   return hugeSize;
}
#endif // NUMA_SHARD

} // namespace mem
