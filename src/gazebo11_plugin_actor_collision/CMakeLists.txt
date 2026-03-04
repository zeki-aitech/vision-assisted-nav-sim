cmake_minimum_required(VERSION 3.5)
project(gazebo11_plugin_actor_collision)

# Suppress CMake dev warning from system Gazebo config (PkgConfig name mismatch)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(ActorCollisionsPlugin SHARED ActorCollisionsPlugin.cc)
target_link_libraries(ActorCollisionsPlugin ${GAZEBO_LIBRARIES})

# Install the plugin library so Gazebo can load it
install(TARGETS ActorCollisionsPlugin
  LIBRARY DESTINATION lib
)

# Environment hook: prepend this package's lib to GAZEBO_PLUGIN_PATH when workspace is sourced
ament_environment_hooks(env-hook/gazebo_plugin_path.sh)

ament_package()
