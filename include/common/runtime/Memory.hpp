#pragma once
#include <stdexcept>
#include <sys/mman.h>

#if defined(__linux__) && defined(NUMA_POOLS) && defined(NUMA_MBIND)
#include <numaif.h>
#endif

namespace mem {

#if defined(__linux__) && defined(NUMA_POOLS) && defined(NUMA_MBIND)
inline void* malloc_huge(size_t size, int numaNode = -1) {
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (p == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
   }

   // Advise kernel to use huge pages
   madvise(p, size, MADV_HUGEPAGE);

   // Bind memory to specific NUMA node if requested
   if (numaNode >= 0) {
      unsigned long nodemask = 1UL << numaNode;
      long result = mbind(p, size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
      if (result != 0) {
         // If mbind fails, just warn but don't abort (memory still usable, just not NUMA-bound)
         // In production you might want to log this
      }
   }

   return p;
}
#else
// Fallback version without NUMA binding
inline void* malloc_huge(size_t size, int numaNode = -1) {
   (void)numaNode; // Suppress unused parameter warning
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef __linux__
   if (p != MAP_FAILED) {
      madvise(p, size, MADV_HUGEPAGE);
   }
#endif
   return p;
}
#endif

inline void free_huge(void* p, size_t size) {
   auto r = munmap(p, size);
   if (r) throw std::runtime_error("Memory unmapping failed.");
}
} // namespace mem
