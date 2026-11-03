#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "../napf.hpp"
#include "threadhelper.hpp"

namespace napf {

namespace nb = nanobind;

using IndexType = unsigned int;

// helper function to get dummy values
template<typename Type>
Type max_and_negative_if_signed() {
  Type max_val = std::numeric_limits<Type>::max();
  if constexpr (std::is_signed_v<Type>) {
    max_val = -max_val;
  }
  return max_val;
}

// Custom callback to find minimum index within radius with zero heap allocation
template<typename DistT, typename IndexT>
struct MinIndexCallback {
  using DistanceType = DistT;
  using IndexType = IndexT;

  const DistanceType radius;
  IndexT min_idx;
  size_t count{0};

  explicit MinIndexCallback(DistanceType r, IndexT initial_idx)
      : radius(r),
        min_idx(initial_idx) {}

  void init() { count = 0; }
  void clear() { count = 0; }
  size_t size() const noexcept { return count; }
  bool empty() const noexcept { return count == 0; }
  bool full() const noexcept { return true; }

  bool addPoint(DistanceType dist, IndexT index) {
    if (dist < radius) {
      if (count == 0 || index < min_idx) {
        min_idx = index;
      }
      ++count;
    }
    return true;
  }

  DistanceType worstDist() const noexcept { return radius; }
  void sort() {}
};

template<typename DataT, unsigned int metric>
class PyKDT {
public:
  using DistT = typename std::
      conditional<std::is_same<DataT, float>::value, float, double>::type;

  using Tree = ArrayTree<DataT, DistT, IndexType, metric>;
  using Cloud = napf::ArrayCloud<DataT, IndexType>;

  int dim_{};
  const unsigned int metric_ = metric;
  size_t leaf_size_{10};
  int nthread_{1};

  nb::object tree_data_;
  const DataT* tree_data_ptr_ = nullptr;
  IndexType datalen_ = 0;
  std::unique_ptr<Cloud> cloud_;
  std::unique_ptr<Tree> tree_;

  PyKDT() = default;

  PyKDT(PyKDT&& other) noexcept = default;

  PyKDT(nb::ndarray<const DataT, nb::c_contig> tree_data) {
    newtree(tree_data, 10, 1);
  }
  PyKDT(nb::ndarray<const DataT, nb::c_contig> tree_data,
        const size_t leaf_size,
        const int nthread) {
    newtree(tree_data, leaf_size, nthread);
  }

  /* builds a new tree and saves it as unique_ptr */
  void newtree(nb::ndarray<const DataT, nb::c_contig> tree_data,
               const size_t leaf_size = 10,
               const int nthread = 1) {
    if (tree_data.ndim() != 2) {
      throw std::invalid_argument("Input tree_data must be a 2D array.");
    }

    dim_ = static_cast<int>(tree_data.shape(1));
    leaf_size_ = leaf_size;
    nthread_ = nthread;

    nanoflann::KDTreeSingleIndexAdaptorParams params(
        leaf_size,
        nanoflann::KDTreeSingleIndexAdaptorFlags::None,
        static_cast<unsigned int>(nthread));

    tree_data_ = nb::cast(tree_data);
    tree_data_ptr_ = tree_data.data();
    datalen_ = static_cast<IndexType>(tree_data.shape(0));

    cloud_ = std::make_unique<Cloud>(tree_data_ptr_,
                                     static_cast<IndexType>(tree_data.size()),
                                     dim_);
    tree_ = std::make_unique<Tree>(dim_, *cloud_, params);
  }

  /* given query points, returns distances and indices */
  nb::tuple knn_search(nb::ndarray<const DataT, nb::c_contig> qpts,
                       const int kneighbors,
                       const int nthread) {
    if (qpts.ndim() != 2 || static_cast<int>(qpts.shape(1)) != dim_) {
      throw std::invalid_argument(
          "Query array must be 2D with the second dimension matching tree dim ("
          + std::to_string(dim_) + ").");
    }
    if (kneighbors <= 0) {
      throw std::invalid_argument("kneighbors must be greater than 0.");
    }
    if (!tree_) {
      throw std::runtime_error("Tree is not initialized.");
    }

    const DataT* q_buf_ptr = qpts.data();
    const size_t qlen = qpts.shape(0);
    const size_t total_elements = qlen * static_cast<size_t>(kneighbors);

    // Allocate raw memory via std::unique_ptr
    auto i_ptr = std::make_unique<IndexType[]>(total_elements);
    auto d_ptr = std::make_unique<DistT[]>(total_elements);

    IndexType* raw_i = i_ptr.get();
    DistT* raw_d = d_ptr.get();

    if (kneighbors > static_cast<int>(datalen_)) {
      std::cout << "WARNING - kneighbors (" << kneighbors
                << ") is bigger than number of tree data (" << datalen_ << ")! "
                << "Tail entries will be filled with dummy values."
                << std::endl;
    }

    const DistT dummy_dist = max_and_negative_if_signed<DistT>();
    const IndexType dummy_index = max_and_negative_if_signed<IndexType>();

    auto searchknn = [&](int begin, int end, int) {
      for (int i{begin}; i < end; i++) {
        const size_t j = static_cast<size_t>(i) * dim_;
        const size_t k = static_cast<size_t>(i) * kneighbors;
        const size_t n_found = tree_->knnSearch(&q_buf_ptr[j],
                                                static_cast<size_t>(kneighbors),
                                                &raw_i[k],
                                                &raw_d[k]);
        for (size_t m = n_found; m < static_cast<size_t>(kneighbors); ++m) {
          raw_i[k + m] = dummy_index;
          raw_d[k + m] = dummy_dist;
        }
      }
    };

    {
      nb::gil_scoped_release release;
      nthread_execution(searchknn, static_cast<int>(qlen), nthread);
    }

    size_t shape[2] = {qlen, static_cast<size_t>(kneighbors)};
    IndexType* final_i = i_ptr.release();
    DistT* final_d = d_ptr.release();

    nb::capsule i_owner(final_i, [](void* p) noexcept {
      delete[] static_cast<IndexType*>(p);
    });
    nb::capsule d_owner(final_d, [](void* p) noexcept {
      delete[] static_cast<DistT*>(p);
    });

    auto dist_arr =
        nb::ndarray<nb::numpy, DistT, nb::shape<2>>(final_d, 2, shape, d_owner);
    auto indices_arr = nb::ndarray<nb::numpy, IndexType, nb::shape<2>>(final_i,
                                                                       2,
                                                                       shape,
                                                                       i_owner);

    return nb::make_tuple(dist_arr, indices_arr);
  }

  /* scipy KDTree style query */
  nb::tuple query(nb::ndarray<const DataT, nb::c_contig> qpts,
                  const int nthread) {
    return knn_search(qpts, 1, nthread);
  }

  /* radius search */
  nb::tuple radius_search(nb::ndarray<const DataT, nb::c_contig> qpts,
                          const DistT radius,
                          const bool return_sorted,
                          const int nthread) {
    if (qpts.ndim() != 2 || static_cast<int>(qpts.shape(1)) != dim_) {
      throw std::invalid_argument(
          "Query array must be 2D with the second dimension matching tree dim ("
          + std::to_string(dim_) + ").");
    }
    if (!tree_) {
      throw std::runtime_error("Tree is not initialized.");
    }

    const DataT* q_buf_ptr = qpts.data();
    const size_t qlen = qpts.shape(0);

    nanoflann::SearchParameters params;
    params.sorted = return_sorted;

    std::vector<std::unique_ptr<IndexType[]>> results_i(qlen);
    std::vector<std::unique_ptr<DistT[]>> results_d(qlen);
    std::vector<size_t> counts(qlen, 0);

    auto searchradius = [&](int begin, int end, int) {
      std::vector<nanoflann::ResultItem<IndexType, DistT>> thread_matches;
      for (int i{begin}; i < end; i++) {
        thread_matches.clear();
        const auto nmatches =
            tree_->radiusSearch(&q_buf_ptr[static_cast<size_t>(i) * dim_],
                                radius,
                                thread_matches,
                                params);
        auto out_i = std::make_unique<IndexType[]>(nmatches);
        auto out_d = std::make_unique<DistT[]>(nmatches);
        for (size_t k = 0; k < nmatches; ++k) {
          out_i[k] = thread_matches[k].first;
          out_d[k] = thread_matches[k].second;
        }
        results_i[i] = std::move(out_i);
        results_d[i] = std::move(out_d);
        counts[i] = nmatches;
      }
    };

    {
      nb::gil_scoped_release release;
      nthread_execution(searchradius, static_cast<int>(qlen), nthread);
    }

    nb::list out_indices;
    nb::list out_dist;

    for (size_t i = 0; i < qlen; ++i) {
      const size_t cnt = counts[i];
      size_t shape[1] = {cnt};
      IndexType* raw_i = results_i[i].release();
      DistT* raw_d = results_d[i].release();

      nb::capsule i_owner(raw_i, [](void* p) noexcept {
        delete[] static_cast<IndexType*>(p);
      });
      nb::capsule d_owner(raw_d, [](void* p) noexcept {
        delete[] static_cast<DistT*>(p);
      });

      auto arr_i = nb::ndarray<nb::numpy, IndexType, nb::shape<1>>(raw_i,
                                                                   1,
                                                                   shape,
                                                                   i_owner);
      auto arr_d =
          nb::ndarray<nb::numpy, DistT, nb::shape<1>>(raw_d, 1, shape, d_owner);

      out_indices.append(arr_i);
      out_dist.append(arr_d);
    }

    return nb::make_tuple(out_indices, out_dist);
  }

  /* radius knn search */
  nb::tuple rknn_search(nb::ndarray<const DataT, nb::c_contig> qpts,
                        const DistT radius,
                        const int n_nearest,
                        const int nthread) {
    if (qpts.ndim() != 2 || static_cast<int>(qpts.shape(1)) != dim_) {
      throw std::invalid_argument(
          "Query array must be 2D with the second dimension matching tree dim ("
          + std::to_string(dim_) + ").");
    }
    if (n_nearest <= 0) {
      throw std::invalid_argument("n_nearest must be greater than 0.");
    }
    if (!tree_) {
      throw std::runtime_error("Tree is not initialized.");
    }

    const DataT* q_buf_ptr = qpts.data();
    const size_t qlen = qpts.shape(0);
    const size_t total_elements = qlen * static_cast<size_t>(n_nearest);

    auto i_ptr = std::make_unique<IndexType[]>(total_elements);
    auto d_ptr = std::make_unique<DistT[]>(total_elements);

    IndexType* raw_i = i_ptr.get();
    DistT* raw_d = d_ptr.get();

    const DistT dummy_dist = max_and_negative_if_signed<DistT>();
    const IndexType dummy_index = max_and_negative_if_signed<IndexType>();

    auto searchradiusknn = [&](int begin, int end, int) {
      for (int i{begin}; i < end; i++) {
        const size_t q_idx = static_cast<size_t>(i) * dim_;
        // Fix buffer offset bug: use n_nearest instead of dim_
        const size_t out_idx = static_cast<size_t>(i) * n_nearest;

        IndexType* cur_i = &raw_i[out_idx];
        DistT* cur_d = &raw_d[out_idx];

        const auto n_matches = tree_->rknnSearch(&q_buf_ptr[q_idx],
                                                 static_cast<size_t>(n_nearest),
                                                 cur_i,
                                                 cur_d,
                                                 radius);

        for (size_t j{n_matches}; j < static_cast<size_t>(n_nearest); ++j) {
          cur_i[j] = dummy_index;
          cur_d[j] = dummy_dist;
        }
      }
    };

    {
      nb::gil_scoped_release release;
      nthread_execution(searchradiusknn, static_cast<int>(qlen), nthread);
    }

    size_t shape[2] = {qlen, static_cast<size_t>(n_nearest)};
    IndexType* final_i = i_ptr.release();
    DistT* final_d = d_ptr.release();

    nb::capsule i_owner(final_i, [](void* p) noexcept {
      delete[] static_cast<IndexType*>(p);
    });
    nb::capsule d_owner(final_d, [](void* p) noexcept {
      delete[] static_cast<DistT*>(p);
    });

    auto indices_arr = nb::ndarray<nb::numpy, IndexType, nb::shape<2>>(final_i,
                                                                       2,
                                                                       shape,
                                                                       i_owner);
    auto dist_arr =
        nb::ndarray<nb::numpy, DistT, nb::shape<2>>(final_d, 2, shape, d_owner);

    return nb::make_tuple(indices_arr, dist_arr);
  }

  nb::list query_ball_point(nb::ndarray<const DataT, nb::c_contig> qpts,
                            const DistT radius,
                            const bool return_sorted,
                            const int nthread) {
    if (qpts.ndim() != 2 || static_cast<int>(qpts.shape(1)) != dim_) {
      throw std::invalid_argument(
          "Query array must be 2D with the second dimension matching tree dim ("
          + std::to_string(dim_) + ").");
    }
    if (!tree_) {
      throw std::runtime_error("Tree is not initialized.");
    }

    const DataT* q_buf_ptr = qpts.data();
    const size_t qlen = qpts.shape(0);

    nanoflann::SearchParameters params;
    params.sorted = false;

    std::vector<std::unique_ptr<IndexType[]>> results_i(qlen);
    std::vector<size_t> counts(qlen, 0);

    auto searchradius = [&](int start, int end, int) {
      std::vector<nanoflann::ResultItem<IndexType, DistT>> thread_matches;
      for (int i{start}; i < end; ++i) {
        thread_matches.clear();
        const auto nmatches =
            tree_->radiusSearch(&q_buf_ptr[static_cast<size_t>(i) * dim_],
                                radius,
                                thread_matches,
                                params);
        auto out_i = std::make_unique<IndexType[]>(nmatches);
        for (size_t k = 0; k < nmatches; ++k) {
          out_i[k] = thread_matches[k].first;
        }
        if (return_sorted) {
          std::sort(out_i.get(), out_i.get() + nmatches);
        }
        results_i[i] = std::move(out_i);
        counts[i] = nmatches;
      }
    };

    {
      nb::gil_scoped_release release;
      nthread_execution(searchradius, static_cast<int>(qlen), nthread);
    }

    nb::list out_indices;
    for (size_t i = 0; i < qlen; ++i) {
      const size_t cnt = counts[i];
      size_t shape[1] = {cnt};
      IndexType* raw_i = results_i[i].release();
      nb::capsule i_owner(raw_i, [](void* p) noexcept {
        delete[] static_cast<IndexType*>(p);
      });
      auto arr_i = nb::ndarray<nb::numpy, IndexType, nb::shape<1>>(raw_i,
                                                                   1,
                                                                   shape,
                                                                   i_owner);
      out_indices.append(arr_i);
    }

    return out_indices;
  }

  nb::tuple tree_data_unique_inverse(const DistT radius,
                                     const bool return_intersection,
                                     const int nthread) {
    if (!tree_ || datalen_ == 0) {
      throw std::runtime_error("Tree is not initialized or empty.");
    }

    const DataT* q_buf_ptr = tree_data_ptr_;
    const size_t qlen = static_cast<size_t>(datalen_);

    nanoflann::SearchParameters params;
    params.sorted = false;

    auto o_inv = std::make_unique<IndexType[]>(qlen);
    IndexType* o_i_ptr = o_inv.get();

    std::vector<std::unique_ptr<IndexType[]>> results_intersection;
    std::vector<size_t> counts;
    if (return_intersection) {
      results_intersection.resize(qlen);
      counts.resize(qlen, 0);
    }

    const size_t index_t_dim = static_cast<size_t>(dim_);

    if (!return_intersection) {
      // Zero heap allocation fast-path using MinIndexCallback
      auto searchradius = [&](int start, int end, int) {
        for (size_t i = static_cast<size_t>(start);
             i < static_cast<size_t>(end);
             ++i) {
          MinIndexCallback<DistT, IndexType> cb(radius,
                                                static_cast<IndexType>(i));
          (void) tree_->radiusSearchCustomCallback(&q_buf_ptr[i * index_t_dim],
                                                   cb,
                                                   params);
          o_i_ptr[i] = cb.min_idx;
        }
      };

      {
        nb::gil_scoped_release release;
        nthread_execution(searchradius, static_cast<int>(qlen), nthread);
      }
    } else {
      // With intersection
      auto searchradius = [&](int start, int end, int) {
        std::vector<nanoflann::ResultItem<IndexType, DistT>> thread_matches;
        for (size_t i = static_cast<size_t>(start);
             i < static_cast<size_t>(end);
             ++i) {
          thread_matches.clear();
          const auto nmatches = tree_->radiusSearch(&q_buf_ptr[i * index_t_dim],
                                                    radius,
                                                    thread_matches,
                                                    params);
          auto out_i = std::make_unique<IndexType[]>(nmatches);
          for (size_t k = 0; k < nmatches; ++k) {
            out_i[k] = thread_matches[k].first;
          }
          std::sort(out_i.get(), out_i.get() + nmatches);
          o_i_ptr[i] = (nmatches > 0) ? out_i[0] : static_cast<IndexType>(i);
          results_intersection[i] = std::move(out_i);
          counts[i] = nmatches;
        }
      };

      {
        nb::gil_scoped_release release;
        nthread_execution(searchradius, static_cast<int>(qlen), nthread);
      }
    }

    size_t shape[1] = {qlen};
    IndexType* final_o_i = o_inv.release();
    nb::capsule o_owner(final_o_i, [](void* p) noexcept {
      delete[] static_cast<IndexType*>(p);
    });
    auto original_inverse =
        nb::ndarray<nb::numpy, IndexType, nb::shape<1>>(final_o_i,
                                                        1,
                                                        shape,
                                                        o_owner);

    nb::list intersection_list;
    if (return_intersection) {
      for (size_t i = 0; i < qlen; ++i) {
        const size_t cnt = counts[i];
        size_t sub_shape[1] = {cnt};
        IndexType* raw_sub = results_intersection[i].release();
        nb::capsule sub_owner(raw_sub, [](void* p) noexcept {
          delete[] static_cast<IndexType*>(p);
        });
        auto sub_arr =
            nb::ndarray<nb::numpy, IndexType, nb::shape<1>>(raw_sub,
                                                            1,
                                                            sub_shape,
                                                            sub_owner);
        intersection_list.append(sub_arr);
      }
    }

    return nb::make_tuple(original_inverse, intersection_list);
  }

  /* radii search. each query can have a different radius */
  nb::tuple radii_search(nb::ndarray<const DataT, nb::c_contig> qpts,
                         nb::ndarray<const DistT, nb::c_contig> radii,
                         const bool return_sorted,
                         const int nthread) {
    if (qpts.ndim() != 2 || static_cast<int>(qpts.shape(1)) != dim_) {
      throw std::invalid_argument(
          "Query array must be 2D with the second dimension matching tree dim ("
          + std::to_string(dim_) + ").");
    }
    const size_t qlen = qpts.shape(0);
    const size_t rlen = radii.size();
    if (qlen != rlen) {
      std::cout << "CRITICAL WARNING - query length (" << qlen
                << ") and radii length (" << rlen << ") differ! "
                << "returning empty tuple." << std::endl;
      return nb::tuple();
    }
    if (!tree_) {
      throw std::runtime_error("Tree is not initialized.");
    }

    const DataT* q_buf_ptr = qpts.data();
    const DistT* r_buf_ptr = radii.data();

    nanoflann::SearchParameters params;
    params.sorted = return_sorted;

    std::vector<std::unique_ptr<IndexType[]>> results_i(qlen);
    std::vector<std::unique_ptr<DistT[]>> results_d(qlen);
    std::vector<size_t> counts(qlen, 0);

    auto searchradius = [&](int start, int end, int) {
      std::vector<nanoflann::ResultItem<IndexType, DistT>> thread_matches;
      // Fixed outer variable bug: per-query result inside loop
      for (int i{start}; i < end; ++i) {
        thread_matches.clear();
        const auto nmatches =
            tree_->radiusSearch(&q_buf_ptr[static_cast<size_t>(i) * dim_],
                                r_buf_ptr[i],
                                thread_matches,
                                params);

        auto out_i = std::make_unique<IndexType[]>(nmatches);
        auto out_d = std::make_unique<DistT[]>(nmatches);
        for (size_t k = 0; k < nmatches; ++k) {
          out_i[k] = thread_matches[k].first;
          out_d[k] = thread_matches[k].second;
        }
        results_i[i] = std::move(out_i);
        results_d[i] = std::move(out_d);
        counts[i] = nmatches;
      }
    };

    {
      nb::gil_scoped_release release;
      nthread_execution(searchradius, static_cast<int>(qlen), nthread);
    }

    nb::list out_indices;
    nb::list out_dist;

    for (size_t i = 0; i < qlen; ++i) {
      const size_t cnt = counts[i];
      size_t shape[1] = {cnt};
      IndexType* raw_i = results_i[i].release();
      DistT* raw_d = results_d[i].release();

      nb::capsule i_owner(raw_i, [](void* p) noexcept {
        delete[] static_cast<IndexType*>(p);
      });
      nb::capsule d_owner(raw_d, [](void* p) noexcept {
        delete[] static_cast<DistT*>(p);
      });

      auto arr_i = nb::ndarray<nb::numpy, IndexType, nb::shape<1>>(raw_i,
                                                                   1,
                                                                   shape,
                                                                   i_owner);
      auto arr_d =
          nb::ndarray<nb::numpy, DistT, nb::shape<1>>(raw_d, 1, shape, d_owner);

      out_indices.append(arr_i);
      out_dist.append(arr_d);
    }

    return nb::make_tuple(out_indices, out_dist);
  }
};

template<typename T, unsigned int metric>
void add_kdt_pyclass(nb::module_& m, const char* class_name) {
  using KDT = PyKDT<T, metric>;

  nb::class_<KDT>(m, class_name)
      .def(nb::init<>())
      .def(nb::init<nb::ndarray<const T, nb::c_contig>, size_t, int>(),
           nb::arg("tree_data"),
           nb::arg("leaf_size") = 10,
           nb::arg("nthread") = 1)
      .def_prop_ro("tree_data", [](const KDT& kdt) { return kdt.tree_data_; })
      .def_ro("dim", &KDT::dim_)
      .def_ro("metric", &KDT::metric_)
      .def("newtree",
           &KDT::newtree,
           nb::arg("tree_data"),
           nb::arg("leaf_size") = 10,
           nb::arg("nthread") = 1)
      .def("knn_search",
           &KDT::knn_search,
           nb::arg("queries"),
           nb::arg("kneighbors"),
           nb::arg("nthread"))
      .def("query", &KDT::query, nb::arg("queries"), nb::arg("nthread"))
      .def("radius_search",
           &KDT::radius_search,
           nb::arg("queries"),
           nb::arg("radius"),
           nb::arg("return_sorted"),
           nb::arg("nthread"))
      .def("rknn_search",
           &KDT::rknn_search,
           nb::arg("queries"),
           nb::arg("radius"),
           nb::arg("n_nearest"),
           nb::arg("nthread"))
      .def("query_ball_point",
           &KDT::query_ball_point,
           nb::arg("queries"),
           nb::arg("radius"),
           nb::arg("return_sorted"),
           nb::arg("nthread"))
      .def("radii_search",
           &KDT::radii_search,
           nb::arg("queries"),
           nb::arg("radii"),
           nb::arg("return_sorted"),
           nb::arg("nthread"))
      .def("tree_data_unique_inverse",
           &KDT::tree_data_unique_inverse,
           nb::arg("radius"),
           nb::arg("return_intersection") = true,
           nb::arg("nthread") = 1);
}

} // namespace napf
