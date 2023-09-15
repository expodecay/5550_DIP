# CMakeList.txt : CMake project for CMakeProject1test, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("CMakeProject1test")

find_package(OpenCV REQUIRED PATHS "C:/OpenCV/build")
find_package(OpenCV REQUIRED PATHS "C:/OpenCV/build/install")

include_directories(${OpenCV_INCLUDE_DIR})

# Add source to this project's executable.
add_executable (CMakeProject1test "CMakeProject1test.cpp" "CMakeProject1test.h")

target_link_libraries(CMakeProject1test ${OpenCV_LIBS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject1test PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
