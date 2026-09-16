#pragma once
#include <stdexcept>
#include <sys/mman.h>
#include <linux/mman.h>  // MAP_HUGE_2MB / MAP_HUGE_1GB (not pulled in by sys/mman.h)

namespace mem {
inline void* malloc_huge(size_t size) {
#if defined(HUGE_2MB_MALLOC_HUGE)
   constexpr size_t PAGE = 2 * 1024 * 1024; // 2MB
   size_t allocSize = (size + PAGE - 1) & ~(PAGE - 1);
   void* p = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
   if (p == MAP_FAILED) {
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

} // namespace mem
