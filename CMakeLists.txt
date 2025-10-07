cmake_minimum_required(VERSION 3.16)
project(image_enhancer CXX)

# Options
option(NO_SUPERRES "Build without OpenCV dnn_superres (opencv_contrib)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Target
add_executable(image_enhancer
	src/main.cpp
)

# Preprocessor flags
if(NO_SUPERRES)
	target_compile_definitions(image_enhancer PRIVATE NO_SUPERRES=1)
endif()

# Include and link OpenCV
TARGET_INCLUDE_DIRECTORIES(image_enhancer PRIVATE ${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(image_enhancer PRIVATE ${OpenCV_LIBS})

# On Windows, set console subsystem explicitly (optional)
if(WIN32)
	set_target_properties(image_enhancer PROPERTIES OUTPUT_NAME "image_enhancer")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "NO_SUPERRES: ${NO_SUPERRES}")
