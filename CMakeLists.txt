cmake_minimum_required (VERSION 2.6 FATAL_ERROR)
project (oni_converter) 

#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories( ${PROJECT_SOURCE_DIR}/include )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

# PCL required
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#If you have errors in make, with vtk
#find_package(PCL 1.7 REQUIRED)

#if(NOT "${PCL_LIBRARIES}" STREQUAL "")
#  list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
#endif()

message("\n\nFound PCL\n\n")

# Opencv required
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

message("\n\nFound OpenCV\n\n")

# Openni2 required
find_package(OpenNI2 REQUIRED)
include_directories( ${OPENNI2_INCLUDE_DIRS} )
add_definitions(${OPENNI2_DEFINITIONS})

add_executable(oni_converter oni_converter.cpp)
# Openni2 required

TARGET_LINK_LIBRARIES(oni_converter ${OPENNI2_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
