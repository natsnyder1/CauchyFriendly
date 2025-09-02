# Main estimator API executable
add_library(CauchyAPI cauchy_estimator_api.cpp)
add_executable(APItest api_test.cpp)
target_link_libraries(CauchyAPI PUBLIC CauchyLibrary)
target_link_libraries(APItest PUBLIC CauchyAPI)
