cmake_minimum_required(VERSION 3.10)  # CMake version check
project(tiny_ising)

# Add main.cpp file of the project root directory as a source file
set(ISING_SOURCE_FILES tiny_ising.c ising.c wtime.c)

set(DEMO_SOURCE_FILES demo.c ising.c wtime.c)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(tiny_ising ${ISING_SOURCE_FILES})

target_link_libraries(tiny_ising m)

find_package(glfw3 3.3 REQUIRED)


add_executable(demo ${DEMO_SOURCE_FILES})
find_package(OpenGL REQUIRED)

target_link_libraries(demo m OpenGL::GL glfw)