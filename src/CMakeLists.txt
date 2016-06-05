cmake_minimum_required(VERSION 2.8.10)

# set project name
project(03_Filter)

# reject msvc10 caused by not supporting cpp11
if(MSVC10 OR MSVC11)
    message(FATAL_ERROR "Please use an newer Visual Studio than 2012! It does not support C++11 which is required for this lab.")
endif()

# input image selection
set(InputImage_DIR ../data/lena.tiff CACHE FILEPATH "Input image file")
add_definitions(-DINPUTIMAGE=\"${InputImage_DIR}\")

# set compile flags
if(CMAKE_COMPILER_IS_GNUCXX)
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 4.7 OR ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL 4.7)
        message(STATUS "C++11 activated.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
    elseif(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 4.3 OR ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL 4.3)
        message(WARNING "C++0x activated. If you get any errors update to a compiler which fully supports C++11")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else()
        message(FATAL_ERROR "C++11 needed. Therefore a gcc compiler with a version higher than 4.3 is needed.")
    endif()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.0)
        message(STATUS "C++11 activated.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        message(FATAL_ERROR "C++11 needed. Therefore a clang compiler with a version higher than 3.0 is needed.")
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")

# find libraries
find_package(OpenCV REQUIRED)
list(APPEND OPENCV_COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_highgui
)

# set include directories
include_directories(${OpenCV_INCLUDE_DIR})

# define source files
list(APPEND SRCS
    main.cpp
    Threshold.cpp
    Histogram.cpp
    PointOperations.cpp
    Filter.cpp
)

# define header files
list(APPEND HDRS
    Timer.h
    Threshold.h
    Histogram.h
    PointOperations.h
    Filter.h
)

# make executable
add_executable(${PROJECT_NAME} ${SRCS} ${HDRS})

# link against libraries
target_link_libraries(${PROJECT_NAME} ${OPENCV_COMPONENTS})

# copy dlls for windows
if(WIN32)
    if(OPENCV_COMPONENTS)
        foreach(OPENCV_COMPONENT ${OPENCV_COMPONENTS})
            list(APPEND COPY_FILES
            $<TARGET_FILE:${OPENCV_COMPONENT}>)
        endforeach()
    endif()
    foreach(COPY_FILE ${COPY_FILES})
        message(STATUS ${PROJECT_NAME})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${COPY_FILE} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endforeach()
endif()
