# Main estimator API executable

# find_package(nlohmann_json REQUIRED)
# better to just include it for shared project
add_library(CauchyAPI cauchy_estimator_api.cpp)
add_executable(APItest api_test.cpp)
target_link_libraries(CauchyAPI PUBLIC CauchyLibrary)
target_link_libraries(CauchyAPI PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(APItest PUBLIC CauchyAPI)
