#include "../pykdt.hpp"

namespace napf {

void init_double_trees(nb::module_& m) {
  add_kdt_pyclass<double, 1>(m, "KDTdL1");
  add_kdt_pyclass<double, 2>(m, "KDTdL2");
}

} // namespace napf
