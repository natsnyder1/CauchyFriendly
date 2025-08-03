# Main CauchyEstimator executable
add_executable(CauchyEstimator cauchy_estimator.cpp)
target_link_libraries(CauchyEstimator PUBLIC CauchyLibrary)

# set(TARGET_APP cauchy_estimator)

# add_executable(${TARGET_APP}
# ${EXT_DIR}/kawasaki/khi_robot_client.cpp
# ${EXT_DIR}/kawasaki/khi_robot_krnx_driver.cpp
# ${SHARED_SRC}/system/Exo.cpp
# ${SHARED_SRC}/system/ExoTimer.cpp
# ${SHARED_SRC}/system/Safety.cpp
# ${SHARED_SRC}/sensor/ForceSensor.cpp
# ${SHARED_SRC}/sensor/AnalogInput.cpp
# ${SHARED_SRC}/robot/Robot.cpp
# ${SHARED_SRC}/robot/RobotDynamics.cpp
# ${SHARED_SRC}/robot/RobotKinematics.cpp
# ${SHARED_SRC}/robot/AdmittanceController.cpp
# ${SHARED_SRC}/data/Unity.cpp
# ${SHARED_SRC}/data/Telemetry.cpp
# ${SHARED_SRC}/data/TelemetryLogger.cpp
# full_body_control.cpp
# )


# ###############################################
# ## Do this later to include all src files
# # file(GLOB_RECURSE PROJECT_SRC_FILES
# # ${SHARED_SRC}/*.cpp 
# # )

# # add_executable(${TARGET_APP}
# #   ${PROJECT_SRC_FILES}
# #   ${EXT_DIR}/kawasaki/khi_robot_client.cpp
# #   ${EXT_DIR}/kawasaki/khi_robot_krnx_driver.cpp
# #   main.cpp
# # )
# #####################################


# ## For KRNX and DAQ and fmt and mujuco
# find_library(LIB_PTHREAD pthread)
# find_library(LIB_ADLINK pci_dask2k64)
# find_package(fmt)
# target_link_libraries(${TARGET_APP}
#   PRIVATE ${LIB_KRNX}
#   PRIVATE ${LIB_PTHREAD}
#   PRIVATE ${LIB_ADLINK}
#   fmt::fmt
#   mujoco::mujoco
#   gtfo)
  

# ## Install to bin folder
# install(TARGETS ${TARGET_APP})

# ## Install KRNX and mujoco to linux lib - Must be done once, then you can comment out. Requires sudo
# # file (INSTALL ${LIB_KRNX}
# # DESTINATION /usr/lib64/)
# # file (INSTALL ${SHARED_SRC}/gtfo/External/mujoco/bin/lib/libmujoco.so.2.3.1
# # DESTINATION /usr/lib64/)