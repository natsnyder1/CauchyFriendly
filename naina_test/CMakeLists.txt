# Main estimator API executable

# find_package(nlohmann_json REQUIRED)
# better to just include the single_include header for shared project
add_executable(naina_test temp.cpp)
target_link_libraries(naina_test PUBLIC CauchyLibrary)
