#pragma once
#include "Util.hpp"
#include "common/Compat.hpp"
#include <atomic>
#include <functional>
#include <new>
#include <vector>

namespace runtime {

class Barrier {
 private:
   const std::size_t threadCount;
   alignas(CACHELINE_SIZE) std::atomic<std::size_t> cntr;
   alignas(CACHELINE_SIZE) std::atomic<uint8_t> round;

 public:
   explicit Barrier(std::size_t threadCount)
       : threadCount(threadCount), cntr(threadCount), round(0) {}

   template <typename F> bool wait(F finalizer) {
      auto prevRound = round.load(); // Must happen before fetch_sub
      auto prev = cntr.fetch_sub(1);
      if (prev == 1) {
         // last thread arrived
         cntr = threadCount;
         auto r = finalizer();
         round++;
         return r;
      } else {
         while (round == prevRound) {
            // wait until barrier is ready for re-use
	    #if defined(__x86_64__) || defined(_M_X64)
            	asm("pause");
            	asm("pause");
            	asm("pause");
	    #elif defined(__arm__) || defined(__aarch64__)
		// asm volatile("yield" ::: "memory"); // ARM equivalent
		// asm volatile("yield" ::: "memory"); // ARM equivalent
		// asm volatile("yield" ::: "memory"); // ARM equivalent
						    
		asm("yield"); // ARM equivalent
		asm("yield"); // ARM equivalent
		asm("yield"); // ARM equivalent
	    #else
		  // Fallback for other architectures
	    #endif
         }
         return false;
      }
   }
   inline bool wait() {
      return wait([]() { return true; });
   }
};

class HierarchicBarrier {
 public:
   HierarchicBarrier* parent;

 private:
   Barrier barrier;

 public:
   static constexpr size_t threadsPerBarrier = 8;
   HierarchicBarrier(size_t nrThreads, HierarchicBarrier* p)
       : parent(p), barrier(nrThreads) {}

   static std::vector<HierarchicBarrier*> create(size_t nrThreads) {
      std::vector<HierarchicBarrier*> result;
      if (nrThreads <= threadsPerBarrier) {
         auto b = compat::aligned_alloc(alignof(HierarchicBarrier),
                                        sizeof(HierarchicBarrier));
         result.push_back(new (b) HierarchicBarrier(nrThreads, nullptr));
         return result;
      }
      // number of barriers with "threadsPerBarrier" threads
      auto nrBarriersFull = nrThreads / threadsPerBarrier;
      // threads that are in the one barrier which has to few threads
      auto threadsInRestBarrier = nrThreads % threadsPerBarrier;
      // overall number of barriers: full barriers + 1 if there are remaining
      // threads
      auto nrBarriers = nrBarriersFull + (threadsInRestBarrier > 0 ? 1 : 0);
      // create barriers in the hierarchy level above this
      auto parentLevel = create(nrBarriers);
      int64_t parent = -1;
      for (size_t i = 0; i < nrBarriersFull; ++i) {
         if (i % threadsPerBarrier == 0) parent++;
         // create fully filled barrier which uses the current parent
         auto b = compat::aligned_alloc(alignof(HierarchicBarrier),
                                        sizeof(HierarchicBarrier));
         result.push_back(
             new (b) HierarchicBarrier(threadsPerBarrier, parentLevel[parent]));
      }
      // create partially filled barrier
      if (threadsInRestBarrier){
         auto b = compat::aligned_alloc(alignof(HierarchicBarrier),
                                        sizeof(HierarchicBarrier));
         result.push_back(new (b) HierarchicBarrier(threadsInRestBarrier,
                                                    parentLevel.back()));
      }
      return result;
   }

   static void destroy(std::vector<HierarchicBarrier*>& these) {
      std::vector<HierarchicBarrier*> parents;
      for (size_t i = 0; i < these.size(); i += threadsPerBarrier)
         parents.push_back(these[i]->parent);
      for (auto barrier : these) free(barrier);
      if (parents.size() > 1)
         destroy(parents);
      else
         free(parents.back());
   }

   template <typename F> bool wait(F finalizer);

   inline bool wait() {
      return wait([]() {});
   }
};

template <typename F> bool HierarchicBarrier::wait(F finalizer) {
   auto p = parent;
   if (parent)
      return barrier.wait([p, finalizer]() { return p->wait(finalizer); });
   else
      return barrier.wait([finalizer]() {
         finalizer();
         return true;
      });
}
}
