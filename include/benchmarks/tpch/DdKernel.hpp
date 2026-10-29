#pragma once
#include <cstdint>
#include <utility>

/// Data-dependency-width microbenchmark kernel (Typer side).
///
/// `dd::Kernel<W, S>::type::run(extendedprice)` computes, per tuple, the same
/// W intermediates as the Tectorwise dd chain in q1_dd_vectorwise, with
/// Ck = ddConstant(k), in wrapping 64-bit arithmetic:
///
///   chained:      v0 = ep * C0,       vk = v(k-1) * Ck + ep
///   independent:  vk = ep * Ck + ep
///
/// and returns sum_k vk (Tectorwise's dd_sink). With a group size D > 1
/// (chained only) the sum takes only the last value of each group of D steps,
/// vk with (k+1) % D == 0 or k == W-1: the values Tectorwise stores when it
/// computes a group as one expression. Typer keeps the intermediates
/// in registers; Tectorwise materializes each into its own vector buffer. W is
/// a compile-time constant (integer-sequence unrolling, no runtime loop over
/// W), so the machine code is straight-line.
///
/// Every vk is linear in ep, so without a barrier the compiler could fold a
/// whole chain into ep * K, or ep * Ck + ep into one multiply. `opaque` hides a
/// value from the optimizer (an empty asm, no instruction), so each multiply
/// and each add is executed as written, as the Tectorwise primitives are.
namespace dd {

/// W-index-dependent distinct constant so neighbouring intermediate values can
/// neither be CSE'd together nor constant-folded away.
constexpr int64_t ddConstant(int k) {
   return 0x9E3779B97F4A7C15LL * (int64_t)(k + 1) +
          0x51ED270BLL * (int64_t)k + 0x2545F4914F6CDD1DLL;
}

inline uint64_t opaque(uint64_t v) {
   asm volatile("" : "+r"(v));
   return v;
}

/// Chained: a true dependency chain, vk needs v(k-1) (multiply + add latency
/// per step); the running sum is a second, independent chain that takes the
/// group-final values (every value for D = 1).
template <int W, int D = 1>
struct DdKernel {
   template <int k>
   static inline void step(uint64_t ep, uint64_t& v, uint64_t& sum) {
      constexpr uint64_t c = (uint64_t)ddConstant(k);
      if constexpr (k == 0) v = opaque(ep * c);
      else v = opaque(opaque(v * c) + ep);
      if constexpr ((k + 1) % D == 0 || k == W - 1) sum += v;
   }

   template <int... ks>
   static inline uint64_t chain(uint64_t ep, std::integer_sequence<int, ks...>) {
      uint64_t v = 0, sum = 0;
      (step<ks>(ep, v, sum), ...);
      return sum;
   }

   static inline int64_t run(int64_t extendedprice) {
      return (int64_t)chain((uint64_t)extendedprice,
                            std::make_integer_sequence<int, W>{});
   }
};

/// Independent: every vk is computed from the column alone, and all W stay
/// live until a single final reduction (the barrier also stops the compiler
/// from folding each into a running sum as it is produced) -> a wide live set
/// -> register pressure and spills once W exceeds the GP registers, the
/// register-file analogue of Tectorwise keeping W buffers live until its sink.
template <int W>
struct DdKernelIndep {
   template <int k>
   static inline uint64_t gen(uint64_t ep) {
      constexpr uint64_t c = (uint64_t)ddConstant(k);
      return opaque(opaque(ep * c) + ep);
   }

   template <int... ks>
   static inline uint64_t reduce(uint64_t ep, std::integer_sequence<int, ks...>) {
      return (gen<ks>(ep) + ...);
   }

   static inline int64_t run(int64_t extendedprice) {
      return (int64_t)reduce((uint64_t)extendedprice,
                             std::make_integer_sequence<int, W>{});
   }
};

// Kernel shape selector: pick the kernel type at compile time from a Shape.
enum class Shape { Chained, Independent };

template <int W, Shape S, int D = 1>
struct Kernel;
template <int W, int D>
struct Kernel<W, Shape::Chained, D> {
   using type = DdKernel<W, D>;
};
template <int W>
struct Kernel<W, Shape::Independent, 1> {
   using type = DdKernelIndep<W>;
};

}  // namespace dd
