# Main estimator API executable

# find_package(nlohmann_json REQUIRED)
# better to just include the single_include header for shared project
add_library(CauchyAPI cauchy_estimator_api.cpp)
add_executable(APItest api_test.cpp)
target_link_libraries(CauchyAPI PUBLIC CauchyLibrary)
target_link_libraries(APItest PUBLIC CauchyAPI)
