cmake_minimum_required(VERSION 3.15)
project(custom_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(custom_engine
  main.cpp
  tensor.cpp
  model.cpp
  ops.cpp
  image_io.cpp
)

target_include_directories(custom_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(custom_engine PRIVATE nlohmann_json::nlohmann_json)

# stb is vendored third-party code; don't let its warnings drown out ours.
set_source_files_properties(image_io.cpp PROPERTIES COMPILE_FLAGS
  "-Wno-unused-but-set-variable -Wno-missing-field-initializers")
