#pragma once
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <sys/mman.h>
#include <linux/mman.h>  // MAP_HUGE_2MB / MAP_HUGE_1GB (not pulled in by sys/mman.h)

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
#include <cerrno>
#include <cstring>
#include <numaif.h>
#include <string>
#endif

namespace mem {
inline void* malloc_huge(size_t size) {
#if defined(HUGE_2MB_MALLOC_HUGE)
   constexpr size_t PAGE = 2 * 1024 * 1024; // 2MB
   size_t allocSize = (size + PAGE - 1) & ~(PAGE - 1);
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
   if (p == MAP_FAILED) {
      static std::atomic_flag warned = ATOMIC_FLAG_INIT;
      if (!warned.test_and_set())
         fprintf(stderr, "malloc_huge: MAP_HUGETLB failed for %zu bytes, "
                         "falling back to normal pages\n", allocSize);
      p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (p == MAP_FAILED)
         throw std::runtime_error("malloc_huge: mmap failed");
   }
#elif defined(NO_HUGE_PAGES)
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   if (p == MAP_FAILED)
      throw std::runtime_error("malloc_huge: mmap failed");
#else
   // Default: use MADV_HUGEPAGE (THP)
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   if (p == MAP_FAILED)
      throw std::runtime_error("malloc_huge: mmap failed");
#ifdef __linux__
   madvise(p, size, MADV_HUGEPAGE);
#endif
#endif
   return p;
}

inline size_t malloc_huge_size(size_t size) {
#if defined(HUGE_2MB_MALLOC_HUGE)
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

/// Synchronously fault all pages via MADV_POPULATE_WRITE so they are
/// resident before the measurement loop.  Best-effort — silently ignored
/// on kernels older than 5.14.
inline void populate_pages(void* p, size_t size) {
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
   madvise(p, size, MADV_POPULATE_WRITE);
#else
   (void)p; (void)size;
#endif
}

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
/// Allocate anonymous memory bound to a NUMA node (NUMA_SHARD / NUMA_ALLOC).
/// Pages come from malloc_huge, so the hugepage options apply as for any
/// other malloc_huge call; the mapping is bound to the node before any page
/// is touched. With one socket there is nothing to bind.
inline void* malloc_numa(size_t size, int node) {
#ifndef CFG_SOCKETS_COUNT
#define CFG_SOCKETS_COUNT 4
#endif
   void* p = malloc_huge(size);
#if CFG_SOCKETS_COUNT == 1
   (void)node;
#else
   unsigned long nodemask = 1UL << node;
   int rc = mbind(p, malloc_huge_size(size), MPOL_BIND, &nodemask,
                  sizeof(nodemask) * 8, MPOL_MF_STRICT);
   if (rc != 0)
      throw std::runtime_error("malloc_numa: mbind to node " +
                               std::to_string(node) + " failed: " +
                               std::string(std::strerror(errno)));
#endif
   return p;
}

/// The mmap size malloc_numa used, for munmap.
inline size_t malloc_numa_size(size_t size) { return malloc_huge_size(size); }
#endif // NUMA_ALLOC || NUMA_SHARD

} // namespace mem
