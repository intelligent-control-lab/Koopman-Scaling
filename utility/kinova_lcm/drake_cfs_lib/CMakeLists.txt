cmake_minimum_required(VERSION 3.10.2)
project(drake_cmake_installed)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Drake
find_package(drake CONFIG REQUIRED)

# LCM
find_package(lcm REQUIRED)
include(${LCM_USE_FILE})

set(SOURCES 
    src/RobotVisualizer/RobotVisualizer.cpp
    src/KinematicTrajetcoryOptimization/KinematicTrajOpt.cpp
    main.cpp
)

add_executable(trajectory_optimization_lib ${SOURCES})
target_link_libraries(trajectory_optimization_lib drake::drake)
target_include_directories(trajectory_optimization_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/KinematicTrajetcoryOptimization
    ${CMAKE_SOURCE_DIR}/include/RobotVisualizer
    ${CMAKE_SOURCE_DIR}/LCM/exlcm
)
