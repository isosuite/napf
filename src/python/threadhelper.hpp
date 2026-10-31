#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace napf {

template<typename Func, typename IndexT>
void nthread_execution(Func& f, const IndexT total, const IndexT nthread) {
  if (total <= 0) {
    return;
  }

  // if nthread == 1, don't even bother creating thread
  if (nthread == 1 || nthread == 0) {
    f(0, total, 0);
    return;
  }

  int n_usable_threads{static_cast<int>(nthread)};

  // negative input looks for hardware_concurrency
  if (nthread < 0) {
    n_usable_threads = std::max(std::thread::hardware_concurrency(), 1u);
  }

  // thread shouldn't exceed total
  n_usable_threads = std::min(static_cast<int>(total), n_usable_threads);
  if (n_usable_threads <= 1) {
    f(0, total, 0);
    return;
  }

  // get chunk size and prepare threads
  const IndexT chunk_size = (total + n_usable_threads - 1) / n_usable_threads;
  std::vector<std::thread> tpool;
  tpool.reserve(n_usable_threads);

  for (int i{0}; i < (n_usable_threads - 1); i++) {
    tpool.emplace_back(std::thread{f, i * chunk_size, (i + 1) * chunk_size, i});
  }
  {
    // last one
    tpool.emplace_back(std::thread{f,
                                   (n_usable_threads - 1) * chunk_size,
                                   total,
                                   n_usable_threads - 1});
  }

  for (auto& t : tpool) {
    t.join();
  }
}

} // namespace napf
