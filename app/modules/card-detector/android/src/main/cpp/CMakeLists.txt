cmake_minimum_required(VERSION 3.22.1)
project(card_detector)

# OpenCV via prefab (from Maven AAR)
find_package(opencv REQUIRED CONFIG)

add_library(
    card_detector
    SHARED
    card_detector_jni.cpp
    # Shared C++ algorithm (relative path from this CMakeLists location)
    ../../../../../cpp/card_detector.cpp
)

target_include_directories(card_detector PRIVATE
    ../../../../../cpp
)

target_link_libraries(card_detector
    opencv::opencv_core
    opencv::opencv_imgproc
    opencv::opencv_imgcodecs
    android
    log
)
