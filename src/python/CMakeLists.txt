set(NAPF_SOURCES classes/int_trees.cpp classes/long_trees.cpp
                 classes/float_trees.cpp classes/double_trees.cpp napf.cpp)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_napf ${NAPF_SOURCES})
target_link_libraries(_napf PRIVATE napf)
target_compile_definitions(_napf PRIVATE $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(_napf PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(_napf PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
endif()

install(
  TARGETS _napf
  DESTINATION napf
  COMPONENT PythonModule)
