# Main estimator API executable

# find_package(nlohmann_json REQUIRED)
# better to just include the single_include header for shared project
add_executable(APItest api_test.cpp)
target_link_libraries(APItest PUBLIC CauchyLibrary)
