cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
project(fftw_addon)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)

add_compile_definitions(NAPI_VERSION=7)

include(FetchContent)

# FFTW3 — download and build from source
FetchContent_Declare(
	fftw3
	URL https://www.fftw.org/fftw-3.3.10.tar.gz
	URL_HASH MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(ENABLE_FLOAT ON CACHE BOOL "" FORCE)
set(DISABLE_FORTRAN ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fftw3)

# Build the addon
add_library(${PROJECT_NAME} SHARED fftw_addon.cpp ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_JS_INC}
	${fftw3_SOURCE_DIR}/api
	${fftw3_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	${CMAKE_JS_LIB}
	fftw3f
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Windows: generate node.lib from .def
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
	execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
		/out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
