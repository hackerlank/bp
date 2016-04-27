project(bp)
set(CMAKE_CXX_FLAGS "-Wall -pedantic -g -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(APP_SRC1 bp.cpp)
add_executable(bp ${APP_SRC1})

set(APP_SRC2 bp_cpp.cpp)
add_executable(bp_cpp ${APP_SRC2})

set(APP_SRC3 bp_1.cpp)
add_executable(bp_1 ${APP_SRC3})
