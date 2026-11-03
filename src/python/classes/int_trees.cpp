#include "../pykdt.hpp"

namespace napf {

void init_int_trees(nb::module_& m) {
  add_kdt_pyclass<int32_t, 1>(m, "KDTiL1");
  add_kdt_pyclass<int32_t, 2>(m, "KDTiL2");
}

} // namespace napf
