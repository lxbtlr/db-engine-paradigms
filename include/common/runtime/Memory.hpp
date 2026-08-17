#pragma once
#include <stdexcept>
#include <sys/mman.h>
#include <linux/mman.h>  // MAP_HUGE_2MB / MAP_HUGE_1GB (not pulled in by sys/mman.h)

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
#include <cerrno>
#include <cstring>
#include <numaif.h>
#endif

namespace mem {
inline void* malloc_huge(size_t size) {
#if defined(HUGE_1GB_MALLOC_HUGE)
   constexpr size_t PAGE = 1UL * 1024 * 1024 * 1024; // 1GB
   size_t allocSize = (size + PAGE - 1) & ~(PAGE - 1);
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                  -1, 0);
   if (p == MAP_FAILED) {
      p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED)
         throw std::runtime_error("malloc_huge: mmap failed");
   }
#elif defined(HUGE_2MB_MALLOC_HUGE)
   constexpr size_t PAGE = 2 * 1024 * 1024; // 2MB
   size_t allocSize = (size + PAGE - 1) & ~(PAGE - 1);
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
   if (p == MAP_FAILED) {
      p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED)
         throw std::runtime_error("malloc_huge: mmap failed");
   }
#elif defined(NO_HUGE_PAGES)
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (p == MAP_FAILED)
      throw std::runtime_error("malloc_huge: mmap failed");
#else
   // Default: use MADV_HUGEPAGE (THP)
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (p == MAP_FAILED)
      throw std::runtime_error("malloc_huge: mmap failed");
#ifdef __linux__
   madvise(p, size, MADV_HUGEPAGE);
#endif
#endif
   return p;
}

inline size_t malloc_huge_size(size_t size) {
#if defined(HUGE_1GB_MALLOC_HUGE)
   constexpr size_t PAGE = 1UL * 1024 * 1024 * 1024;
   return (size + PAGE - 1) & ~(PAGE - 1);
#elif defined(HUGE_2MB_MALLOC_HUGE)
   constexpr size_t PAGE = 2 * 1024 * 1024;
   return (size + PAGE - 1) & ~(PAGE - 1);
#else
   return size;
#endif
}

inline void free_huge(void* p, size_t size) {
   auto r = munmap(p, malloc_huge_size(size));
   if (r) throw std::runtime_error("Memory unmapping failed.");
}

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
/// Allocate anonymous memory bound to a specific NUMA node.
/// Tries MAP_HUGETLB first; falls back to base pages with MADV_HUGEPAGE.
inline void* malloc_numa(size_t size, int node) {
#ifdef HUGE_1GB_MALLOC_NUMA
   constexpr size_t PAGE = 1UL * 1024 * 1024 * 1024; // 1GB
   constexpr int HUGETLB_FLAGS = MAP_HUGETLB | MAP_HUGE_1GB;
#else
   constexpr size_t PAGE = 2 * 1024 * 1024; // 2MB
   constexpr int HUGETLB_FLAGS = MAP_HUGETLB;
#endif
   size_t allocSize = (size + PAGE - 1) & ~(PAGE - 1);

   // Try hugetlb first (same allocSize either way for consistent cleanup)
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | HUGETLB_FLAGS, -1, 0);
   if (p == MAP_FAILED) {
      // Fall back to base pages, keep rounded size
      p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED)
         throw std::runtime_error("malloc_numa: mmap failed: " +
                                  std::string(std::strerror(errno)));
      madvise(p, allocSize, MADV_HUGEPAGE);
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
#ifdef HUGE_1GB_MALLOC_NUMA
   constexpr size_t PAGE = 1UL * 1024 * 1024 * 1024;
#else
   constexpr size_t PAGE = 2 * 1024 * 1024;
#endif
   return (size + PAGE - 1) & ~(PAGE - 1);
}
#endif // NUMA_ALLOC || NUMA_SHARD

} // namespace mem
