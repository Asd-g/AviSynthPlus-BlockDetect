cmake_minimum_required(VERSION 3.16)

project(BlockDetect LANGUAGES CXX)

set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_sse2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx512.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VCL2/instrset_detect.cpp
)

if (WIN32)
    set(sources ${sources} ${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect.rc)
endif()

add_library(BlockDetect SHARED ${sources})

target_include_directories(BlockDetect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (UNIX)
    target_include_directories(BlockDetect PRIVATE /usr/local/include/avisynth)
else()
    target_include_directories(BlockDetect PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../AviSynthPlus/avs_core/include")
endif()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()

    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL Debug)
        target_compile_definitions(BlockDetect PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL Release)
        target_compile_definitions(BlockDetect PRIVATE RELEASE_BUILD)
    endif()
    
    target_compile_options(BlockDetect PRIVATE $<$<CONFIG:Release>:-s>)

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

target_compile_features(BlockDetect PRIVATE cxx_std_17)

if (MSVC)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_sse2.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse2")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/blockdetect_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")
endif()

target_link_libraries(BlockDetect PRIVATE avisynth)

if (MINGW)
    set_target_properties(BlockDetect PROPERTIES PREFIX "")

    target_link_libraries(BlockDetect PRIVATE -static-libstdc++ -static -lwinpthread)
endif()

if (UNIX)
    find_package (Git)

    if (GIT_FOUND)
        execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(BlockDetect PROPERTIES OUTPUT_NAME "blockdetect.${ver}")
    else()
        message (STATUS "GIT not found")
    endif()

    include(GNUInstallDirs)

    INSTALL(TARGETS BlockDetect LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

    # uninstall target
    if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif()
endif()
