// run_compressbench: does a masked store with an all-zero mask touch memory?
// Checks vpcompressd with a memory destination (_mm256_mask_compressstoreu_epi32,
// as used by the old *_avx512 selection kernels) against the plain masked store
// (_mm256_mask_storeu_epi32) and the register-form compress + masked store
// (VW_SIMD_SEL_COMPRESS=reg). Each variant runs with mask 0 and a nonzero mask.
//
// Tests (-t all|fault|pagefault|cache|tput|align|loop):
//   fault      destination on a PROT_NONE page: does the store fault?
//   pagefault  destination on fresh, never-touched pages: are pages faulted in
//              (minor faults, mincore residency)?
//   cache      destination line flushed first: is it in cache afterwards
//              (latency of a load right after the store)?
//   tput       cycles per store, destination in L1
//   align      cycles per store by destination offset within a 64-byte line
//              (0..60, step 4) and across a 4 KiB page boundary, for masks
//              writing 1, 4 and 8 lanes: which placements split a line or page
//   loop       one variant (-v) in a tight loop, for perf stat, e.g.
//              perf stat -e mem_inst_retired.all_stores,mem_inst_retired.stlb_miss_stores
//                   run_compressbench -t loop -v cmem -k 0
//
// Architecturally, masked stores suppress faults on masked-off elements, and
// vpcompressd writes only popcount(mask) elements; the tests show whether the
// hardware still issues a store (RFO, page walk, microcode assist) for mask 0.
#include <algorithm>
#include <chrono>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#if !defined(__x86_64__) || !defined(__AVX512F__) || !defined(__AVX512VL__)
int main() {
   std::fprintf(stderr, "run_compressbench: needs x86-64 with AVX-512F/VL (build with -march=<avx512 cpu>)\n");
   return 2;
}
#else
#include <immintrin.h>
#include <x86intrin.h>

namespace {

// ------------------------------------------------------------ store variants
// noinline with the mask as a runtime argument, so the compiler cannot fold a
// constant zero mask away.
using StoreFn = void (*)(uint32_t*, __mmask8, __m256i);

__attribute__((noinline)) void st_cmem(uint32_t* p, __mmask8 m, __m256i v) {
   _mm256_mask_compressstoreu_epi32(p, m, v); // vpcompressd ymm -> mem {k}
}
__attribute__((noinline)) void st_masked(uint32_t* p, __mmask8 m, __m256i v) {
   _mm256_mask_storeu_epi32(p, m, v); // vmovdqu32 ymm -> mem {k}
}
__attribute__((noinline)) void st_creg(uint32_t* p, __mmask8 m, __m256i v) {
   // register compress, then store popcount(m) lanes (VW_SIMD_SEL_COMPRESS=reg)
   __m256i packed = _mm256_maskz_compress_epi32(m, v);
   __mmask8 st = (__mmask8)((1u << __builtin_popcount(m)) - 1);
   _mm256_mask_storeu_epi32(p, st, packed);
}
__attribute__((noinline)) void st_none(uint32_t*, __mmask8, __m256i) {}
__attribute__((noinline)) void st_plain(uint32_t* p, __mmask8, __m256i v) {
   _mm256_storeu_si256((__m256i*)p, v); // unmasked control
}

struct Variant {
   const char* name;
   StoreFn fn;
};
const Variant kVariants[] = {{"cmem", st_cmem},   {"masked", st_masked}, {"creg", st_creg},
                             {"plain", st_plain}, {"none", st_none}};

const Variant* findVariant(const std::string& n) {
   for (auto& v : kVariants)
      if (n == v.name) return &v;
   return nullptr;
}

volatile uint32_t sink;
const __m256i kVals = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
const size_t kPage = 4096;

long minflt() {
   rusage u;
   getrusage(RUSAGE_SELF, &u);
   return u.ru_minflt;
}

// ------------------------------------------------------------------- fault
sigjmp_buf jb;
void onSegv(int) { siglongjmp(jb, 1); }

void testFault() {
   std::printf("\n# fault: destination is the first byte of a PROT_NONE page\n");
   std::printf("variant,mask,result\n");
   auto* base = (uint8_t*)mmap(nullptr, 2 * kPage, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   mprotect(base + kPage, kPage, PROT_NONE);
   auto* dst = (uint32_t*)(base + kPage);
   struct sigaction sa {}, old{};
   sa.sa_handler = onSegv;
   sigaction(SIGSEGV, &sa, &old);
   // loop state is volatile: it must survive siglongjmp back into the loop
   const size_t nv = sizeof(kVariants) / sizeof(kVariants[0]);
   for (volatile size_t vi = 0; vi < nv; vi = vi + 1) {
      const Variant& v = kVariants[vi];
      if (v.fn == st_none) continue;
      for (volatile unsigned mi = 0; mi < 2; mi = mi + 1) {
         const __mmask8 m = (__mmask8)mi; // 0x00, then 0x01
         bool faulted = sigsetjmp(jb, 1) != 0;
         if (!faulted) v.fn(dst, m, kVals);
         std::printf("%s,0x%02x,%s\n", v.name, (unsigned)m, faulted ? "SIGSEGV" : "no fault");
      }
   }
   sigaction(SIGSEGV, &old, nullptr);
   munmap(base, 2 * kPage);
}

// --------------------------------------------------------------- pagefault
void testPagefault(size_t pages) {
   std::printf("\n# pagefault: one store per fresh, never-touched 4 KiB page (%zu pages, THP off)\n", pages);
   std::printf("variant,mask,minor_faults,resident_pages\n");
   for (auto& v : kVariants) {
      for (__mmask8 m : {(__mmask8)0x00, (__mmask8)0x01}) {
         if (v.fn == st_none && m) continue;
         size_t len = pages * kPage;
         auto* base = (uint8_t*)mmap(nullptr, len, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
         madvise(base, len, MADV_NOHUGEPAGE);
         long f0 = minflt();
         for (size_t i = 0; i < pages; ++i) v.fn((uint32_t*)(base + i * kPage), m, kVals);
         long f1 = minflt();
         std::vector<unsigned char> res(pages);
         mincore(base, len, res.data());
         size_t resident = 0;
         for (auto r : res) resident += r & 1;
         std::printf("%s,0x%02x,%ld,%zu\n", v.name, m, f1 - f0, resident);
         munmap(base, len);
      }
   }
}

// ------------------------------------------------------------------- cache
inline uint64_t timeLoad(const volatile uint32_t* p) {
   unsigned aux;
   _mm_mfence();
   _mm_lfence();
   uint64_t t0 = __rdtscp(&aux);
   sink = *p;
   uint64_t t1 = __rdtscp(&aux);
   _mm_lfence();
   return t1 - t0;
}

void testCache(size_t lines) {
   std::printf("\n# cache: clflush line, store, then time one load of that line (%zu lines, one per page, random order)\n", lines);
   std::printf("# low cycles = the store brought the line into cache (RFO); high = it stayed in memory\n");
   std::printf("variant,mask,median_cycles,p10_cycles,p90_cycles\n");
   size_t len = lines * kPage;
   auto* base = (uint8_t*)mmap(nullptr, len, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
   std::memset(base, 1, len);
   std::vector<size_t> order(lines);
   for (size_t i = 0; i < lines; ++i) order[i] = i;
   srand(7);
   for (size_t i = lines - 1; i > 0; --i) std::swap(order[i], order[rand() % (i + 1)]);
   for (auto& v : kVariants) {
      for (__mmask8 m : {(__mmask8)0x00, (__mmask8)0xFF}) {
         if (v.fn == st_none && m) continue;
         std::vector<uint64_t> t;
         t.reserve(lines);
         for (size_t rep = 0; rep < 3; ++rep)
            for (size_t i : order) {
               // line in the middle of the page, away from page-start prefetch
               auto* p = (uint32_t*)(base + i * kPage + 1024 + 64 * (rep + 1));
               _mm_clflush(p);
               _mm_mfence();
               v.fn(p, m, kVals);
               t.push_back(timeLoad(p));
            }
         std::sort(t.begin(), t.end());
         std::printf("%s,0x%02x,%lu,%lu,%lu\n", v.name, m, (unsigned long)t[t.size() / 2],
                     (unsigned long)t[t.size() / 10], (unsigned long)t[t.size() * 9 / 10]);
      }
   }
   munmap(base, len);
}

// -------------------------------------------------------------------- tput
double cyclesPerOp(StoreFn fn, __mmask8 m, size_t iters) {
   alignas(64) static uint32_t buf[64 * 16];
   uint64_t best = ~0ull;
   for (int trial = 0; trial < 5; ++trial) {
      uint64_t t0 = __rdtsc();
      for (size_t i = 0; i < iters; ++i) {
         fn(buf + (i & 63) * 16, m, kVals);
         asm volatile("" ::: "memory");
      }
      uint64_t t1 = __rdtsc();
      best = std::min(best, t1 - t0);
   }
   return double(best) / double(iters);
}

void testTput(size_t iters) {
   std::printf("\n# tput: TSC cycles per call (noinline call included), destination in L1, best of 5\n");
   std::printf("variant,mask,tsc_cycles_per_op,minus_none\n");
   double base = cyclesPerOp(st_none, 0, iters);
   for (auto& v : kVariants)
      for (__mmask8 m : {(__mmask8)0x00, (__mmask8)0x0F, (__mmask8)0xFF}) {
         if (v.fn == st_none && m) continue;
         double c = cyclesPerOp(v.fn, m, iters);
         std::printf("%s,0x%02x,%.2f,%.2f\n", v.name, m, c, c - base);
      }
}

// ------------------------------------------------------------------- align
// Destination = line start + off (lines rotate over 16 lines, or over 16
// pages when pageCross). A store writes popcount(mask)*4 bytes from there
// (cmem/creg) or the masked lanes of 32 bytes (masked/plain).
double cyclesPerOpAt(StoreFn fn, __mmask8 m, size_t off, bool pageCross, size_t iters) {
   static uint8_t* region = nullptr;
   if (!region) {
      region = (uint8_t*)mmap(nullptr, 17 * kPage, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
      std::memset(region, 0, 17 * kPage);
   }
   uint32_t* dst[16];
   for (size_t j = 0; j < 16; ++j)
      dst[j] = (uint32_t*)(pageCross ? region + (j + 1) * kPage - 64 + off
                                     : region + 64 * j + off);
   uint64_t best = ~0ull;
   for (int trial = 0; trial < 5; ++trial) {
      uint64_t t0 = __rdtsc();
      for (size_t i = 0; i < iters; ++i) {
         fn(dst[i & 15], m, kVals);
         asm volatile("" ::: "memory");
      }
      uint64_t t1 = __rdtsc();
      best = std::min(best, t1 - t0);
   }
   return double(best) / double(iters);
}

void testAlign(size_t iters) {
   std::printf("\n# align: TSC cycles per call by destination offset (bytes from a 64-byte line start)\n");
   std::printf("# bytes written: mask 0x01 = 4, 0x0f = 16, 0xff = 32 (cmem/creg: from off; masked/plain: lanes of off..off+31)\n");
   std::printf("# page=1: the line is the last one of a 4 KiB page, so off+bytes > 64 also crosses a page\n");
   std::printf("# written_cross: the bytes actually written cross the line end;\n");
   std::printf("# span_cross: the full 32-byte vector off..off+31 crosses it (masked-off lanes included)\n");
   std::printf("variant,mask,page,offset,written_cross,span_cross,tsc_cycles_per_op\n");
   for (auto& v : kVariants) {
      if (v.fn == st_none) continue;
      for (__mmask8 m : {(__mmask8)0x01, (__mmask8)0x0F, (__mmask8)0xFF}) {
         for (int page = 0; page < 2; ++page)
            for (size_t off = 0; off < 64; off += 4) {
               // bytes actually written: compress writes popcount lanes from
               // off; masked writes up to its highest set lane; plain all 32
               size_t bytes = (v.fn == st_cmem || v.fn == st_creg)
                                  ? 4 * __builtin_popcount(m)
                                  : (v.fn == st_plain ? 32 : 4 * (32 - __builtin_clz((unsigned)m)));
               int written = off + bytes > 64, span = off + 32 > 64;
               std::printf("%s,0x%02x,%d,%zu,%d,%d,%.2f\n", v.name, m, page, off, written, span,
                           cyclesPerOpAt(v.fn, m, off, page, iters));
            }
      }
   }
}

// -------------------------------------------------------------------- loop
void testLoop(const Variant& v, __mmask8 m, size_t iters) {
   alignas(64) static uint32_t buf[64 * 16];
   auto t0 = std::chrono::steady_clock::now();
   for (size_t i = 0; i < iters; ++i) {
      v.fn(buf + (i & 63) * 16, m, kVals);
      asm volatile("" ::: "memory");
   }
   auto t1 = std::chrono::steady_clock::now();
   std::printf("loop %s mask=0x%02x iters=%zu ns/op=%.3f\n", v.name, m, iters,
               std::chrono::duration<double, std::nano>(t1 - t0).count() / double(iters));
}

} // namespace

int main(int argc, char** argv) {
   std::string test = "all", variant = "cmem";
   unsigned mask = 0;
   size_t iters = 100000000, pages = 4096, lines = 4096;
   int opt;
   while ((opt = getopt(argc, argv, "t:v:k:n:p:l:")) != -1) {
      switch (opt) {
      case 't': test = optarg; break;
      case 'v': variant = optarg; break;
      case 'k': mask = std::strtoul(optarg, nullptr, 0); break;
      case 'n': iters = std::strtoull(optarg, nullptr, 10); break;
      case 'p': pages = std::strtoull(optarg, nullptr, 10); break;
      case 'l': lines = std::strtoull(optarg, nullptr, 10); break;
      default:
         std::fprintf(stderr,
                      "usage: %s [-t all|fault|pagefault|cache|tput|align|loop] [-v cmem|masked|creg|plain|none]\n"
                      "          [-k mask] [-n iters] [-p pages] [-l lines]\n",
                      argv[0]);
         return 1;
      }
   }
   bool all = test == "all";
   if (all || test == "fault") testFault();
   if (all || test == "pagefault") testPagefault(pages);
   if (all || test == "cache") testCache(lines);
   if (all || test == "tput") testTput(iters / 10);
   if (all || test == "align") testAlign(iters / 100);
   if (test == "loop") {
      auto* v = findVariant(variant);
      if (!v) { std::fprintf(stderr, "unknown variant %s\n", variant.c_str()); return 1; }
      testLoop(*v, (__mmask8)mask, iters);
   }
   return 0;
}
#endif
