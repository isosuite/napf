#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace napf {

void init_int_trees(nb::module_&);
void init_long_trees(nb::module_&);
void init_float_trees(nb::module_&);
void init_double_trees(nb::module_&);

} // namespace napf

NB_MODULE(_napf, m) {
  napf::init_int_trees(m);
  napf::init_long_trees(m);
  napf::init_float_trees(m);
  napf::init_double_trees(m);
}
