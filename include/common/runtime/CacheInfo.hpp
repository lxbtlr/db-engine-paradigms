#pragma once
//===----------------------------------------------------------------------===//
// Cache sizes of the machine the engine runs on, for decisions that depend on
// what fits in cache (VW_JOIN_DISPATCH's Bloom filter rule). Read once from
// Linux sysfs (cpu0, data or unified caches), else sysconf; an environment
// variable VW_CACHE_L<level>_BYTES overrides (for experiments). 0 = unknown.
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace runtime {

namespace detail {
inline size_t readCacheBytes(unsigned level) {
   char env[32];
   std::snprintf(env, sizeof env, "VW_CACHE_L%u_BYTES", level);
   if (const char* v = std::getenv(env)) return std::strtoull(v, nullptr, 10);
   for (unsigned idx = 0; idx < 16; ++idx) {
      const std::string dir =
          "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/";
      auto read = [&](const char* f, char* buf, size_t n) {
         FILE* fp = std::fopen((dir + f).c_str(), "r");
         if (!fp) return false;
         const bool ok = std::fgets(buf, int(n), fp) != nullptr;
         std::fclose(fp);
         return ok;
      };
      char lvl[16], type[32], size[32];
      if (!read("level", lvl, sizeof lvl)) break;
      if (unsigned(std::atoi(lvl)) != level) continue;
      if (!read("type", type, sizeof type) || std::strncmp(type, "Instruction", 11) == 0)
         continue;
      if (!read("size", size, sizeof size)) continue;
      char* end = nullptr;
      size_t bytes = std::strtoull(size, &end, 10);
      if (end && (*end == 'K' || *end == 'k')) bytes <<= 10;
      else if (end && (*end == 'M' || *end == 'm')) bytes <<= 20;
      return bytes;
   }
#if defined(_SC_LEVEL1_DCACHE_SIZE) && defined(_SC_LEVEL2_CACHE_SIZE) &&       \
    defined(_SC_LEVEL3_CACHE_SIZE)
   const long v = level == 1   ? sysconf(_SC_LEVEL1_DCACHE_SIZE)
                  : level == 2 ? sysconf(_SC_LEVEL2_CACHE_SIZE)
                  : level == 3 ? sysconf(_SC_LEVEL3_CACHE_SIZE)
                               : 0;
   if (v > 0) return size_t(v);
#endif
   return 0;
}
} // namespace detail

/// bytes of the level-`level` data (or unified) cache of cpu0; 0 = unknown
inline size_t cacheBytes(unsigned level) {
   static const size_t l1 = detail::readCacheBytes(1);
   static const size_t l2 = detail::readCacheBytes(2);
   static const size_t l3 = detail::readCacheBytes(3);
   return level == 1 ? l1 : level == 2 ? l2 : level == 3 ? l3 : 0;
}

} // namespace runtime
