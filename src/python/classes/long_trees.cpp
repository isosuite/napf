#include "../pykdt.hpp"

namespace napf {

void init_long_trees(nb::module_& m) {
  add_kdt_pyclass<int64_t, 1>(m, "KDTlL1");
  add_kdt_pyclass<int64_t, 2>(m, "KDTlL2");
}

} // namespace napf
