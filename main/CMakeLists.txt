# Main CauchyEstimator executable
add_executable(CauchyEstimator cauchy_estimator.cpp)
target_link_libraries(CauchyEstimator PUBLIC CauchyLibrary)
