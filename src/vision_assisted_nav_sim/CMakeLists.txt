cmake_minimum_required(VERSION 3.8)
project(vision_assisted_nav_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

# install human model for Gazebo
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
