cmake_minimum_required(VERSION 3.12)
project(
  napf
  VERSION 0.1.0
  LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

endif()

# options
option(NAPF_BUILD_PYTHON "build python module" ON)

# config
set(exe_dest "bin")
set(incl_dest "include")
set(lib_dest "lib")
set(cfg_dest "${lib_dest}/cmake/${PROJECT_NAME}")
set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# sources
set(CXX_HEADERS src/napf.hpp)

# Interface Library, since it's header only lib
add_library(napf INTERFACE)
add_library(napf::napf ALIAS napf)

# basic path
target_include_directories(
  napf INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                 $<INSTALL_INTERFACE:${incl_dest}>)

# try to use installed nanoflann. else, include the one in the third_party
find_package(nanoflann QUIET)
if(nanoflann_FOUND AND nanoflann_VERSION VERSION_GREATER_EQUAL "1.5.0")
  message("nanoflann found - napf will link to nanoflann found in system.")
  target_link_libraries(napf INTERFACE nanoflann::nanoflann)
else()
  set(CXX_HEADERS ${CXX_HEADERS} third_party/nanoflann.hpp)
  target_include_directories(
    napf INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party>)
endif()
target_compile_features(napf INTERFACE cxx_std_17)

if(NAPF_BUILD_PYTHON)
  add_subdirectory(src/python)
endif()

# configure config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${version_config}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file("cmake/config.cmake.in" "${project_config}"
                              INSTALL_DESTINATION "${cfg_dest}")

install(
  TARGETS napf
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION ${lib_dest}
  ARCHIVE DESTINATION ${lib_dest}
  INCLUDES
  DESTINATION "${incl_dest}")
install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")

install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")
install(FILES ${CXX_HEADERS} DESTINATION ${incl_dest})
