#####################################################################################################################################################
# CMAKE List
#
# Tips:
#   IF STATEMENT EXAMPLE:
#       set(SRCS src.cpp)
#       if(WIN32)
#           set(SRCS ${SRCS} win_src.cpp)
#       elseif(UNIX)
#           set(SRCS ${SRCS} unix_src.cpp)
#       else()
#
#       endif()
#
#   Make with Debug (Default)
#       $ cd <path_to_cmakelist.txt>
#       $ mkdir build/debug
#       $ cd build/debug
#       $ cmake ../../ -DCMAKE_BUILD_TYPE=Debug
#
#   Make with Release
#       $ cd <path_to_cmakelist.txt>
#       $ mkdir build/release
#       $ cd build/release
#       $ cmake ../../ -DCMAKE_BUILD_TYPE=Release
#
#####################################################################################################################################################

# BEGIN CMAKE Version (Required Line) ---------------------------------------------------------------------------------------------------------------
cmake_minimum_required (VERSION 3.3.2)
# END CMAKE Version ---------------------------------------------------------------------------------------------------------------------------------


# BEGIN Target Name ---------------------------------------------------------------------------------------------------------------------------------
SET(TARGET_NAME espressoTester)
# END Target Name -----------------------------------------------------------------------------------------------------------------------------------


# BEGIN Project Name --------------------------------------------------------------------------------------------------------------------------------
PROJECT(${TARGET_NAME})
# END Project Name-----------------------------------------------------------------------------------------------------------------------------------


# BEGIN Set VS startup Project (Affects Windows only) -----------------------------------------------------------------------------------------------
#SET_PROPERTY(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
# END Set VS startup Project ------------------------------------------------------------------------------------------------------------------------



# BEGIN Compiler Definitions ------------------------------------------------------------------------------------------------------------------------
#add_definitions(-DLINUX)
# END Compiler Definitions --------------------------------------------------------------------------------------------------------------------------


# BEGIN External CMake files ------------------------------------------------------------------------------------------------------------------------
# Make sure include and lib directories are relative to home dir of this cmake file
#ADD_SUBDIRECTORY()
# END External CMake files --------------------------------------------------------------------------------------------------------------------------


# BEGIN Target Sources ------------------------------------------------------------------------------------------------------------------------------
FILE(GLOB TARGET_SRCS
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
)
# END Target Sources --------------------------------------------------------------------------------------------------------------------------------


# BEGIN Target Headers ------------------------------------------------------------------------------------------------------------------------------
FILE(GLOB TARGET_HEADERS
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.cuh
)
# END Target Sources --------------------------------------------------------------------------------------------------------------------------------


# BEGIN Library Settings ----------------------------------------------------------------------------------------------------------------------------
# For Visual studio this call must be before ADD_EXECUTABLE
#link_directories(
#    ${PROJECT_SOURCE_DIR}/../build/Debug/
#)
# END Library Settings ------------------------------------------------------------------------------------------------------------------------------


# BEGIN Target Settings -----------------------------------------------------------------------------------------------------------------------------
ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_SRCS} ${TARGET_HEADERS})
# END Target Settings -------------------------------------------------------------------------------------------------------------------------------


# BEGIN Set Output Path for libs or bins ------------------------------------------------------------------------------------------------------------
SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY_DEBUG
                        ${PROJECT_SOURCE_DIR}/build/debug)
SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY_RELEASE
                        ${PROJECT_SOURCE_DIR}/build/release)

SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        LIBRARY_OUTPUT_DIRECTORY_DEBUG
                        ${PROJECT_SOURCE_DIR}/build/debug)
SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        LIBRARY_OUTPUT_DIRECTORY_RELEASE
                        ${PROJECT_SOURCE_DIR}/build/release)

SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY_DEBUG
                        ${PROJECT_SOURCE_DIR}/build/debug)
SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY_RELEASE
                        ${PROJECT_SOURCE_DIR}/build/release)
# END Set Output Path for libs or bins --------------------------------------------------------------------------------------------------------------


# BEGIN Build dependencies --------------------------------------------------------------------------------------------------------------------------
ADD_DEPENDENCIES(${TARGET_NAME} espresso)
ADD_DEPENDENCIES(${TARGET_NAME} sysc_fpga_shim)
ADD_DEPENDENCIES(${TARGET_NAME} network)
ADD_DEPENDENCIES(${TARGET_NAME} myNetProto)
ADD_DEPENDENCIES(${TARGET_NAME} caffeDataParser)
ADD_DEPENDENCIES(${TARGET_NAME} fixedPoint)
# ADD_DEPENDENCIES(${TARGET_NAME} darknet_arcv)
# ADD_DEPENDENCIES(${TARGET_NAME} util)
# ADD_DEPENDENCIES(${TARGET_NAME} caffeClassifier)
#add_dependencies(${TARGET_NAME} RL)
# END Build dependencies ----------------------------------------------------------------------------------------------------------------------------


# BEGIN Debug / Release Settings --------------------------------------------------------------------------------------------------------------------
IF(EXISTS "${PROJECT_SOURCE_DIR}/DEBUG_BUILD")
    SET(CMAKE_BUILD_TYPE Debug)
	SET(BUILD_DIR_SUFFIX "debug")
ELSEIF(EXISTS "${PROJECT_SOURCE_DIR}/RELEASE_BUILD")
    SET(CMAKE_BUILD_TYPE Release)
	SET(BUILD_DIR_SUFFIX "release")
ELSEIF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
	SET(BUILD_DIR_SUFFIX "debug")
ENDIF()
IF(WIN32)
	SET(CMAKE_CXX_FLAGS_DEBUG "/ZI /Od")
ELSE()
	SET(CMAKE_CXX_FLAGS_DEBUG "-g")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
MESSAGE("[" ${TARGET_NAME} "] Build Type: " ${CMAKE_BUILD_TYPE})
# END Debug / Release Settings ----------------------------------------------------------------------------------------------------------------------


# BEGIN Compiler Flags ------------------------------------------------------------------------------------------------------------------------------
TARGET_COMPILE_OPTIONS(${TARGET_NAME} PUBLIC
    "-std=c++1y"
    "-Wall"
    "-fPIC"
    "-fexceptions"
)
# END Compiler Flags --------------------------------------------------------------------------------------------------------------------------------


# BEGIN Include directories -------------------------------------------------------------------------------------------------------------------------
TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/inc/
    ${PROJECT_SOURCE_DIR}/../caffeDataParser/inc/
    ${PROJECT_SOURCE_DIR}/../espresso/inc/
    ${PROJECT_SOURCE_DIR}/../espresso/inc/CPU
    ${PROJECT_SOURCE_DIR}/../espresso/inc/FPGA
    ${PROJECT_SOURCE_DIR}/../espresso/inc/Darknet
    ${PROJECT_SOURCE_DIR}/../fixedPoint/inc/
    ${PROJECT_SOURCE_DIR}/../caffeClassifier/inc/
    ${PROJECT_SOURCE_DIR}/../darknet/include/
    ${PROJECT_SOURCE_DIR}/../darknet/src/
    ${PROJECT_SOURCE_DIR}/../util/inc/
    ${PROJECT_SOURCE_DIR}/../FPGA_shim/inc/
    ${PROJECT_SOURCE_DIR}/../SYSC_FPGA_shim/inc/
    ${PROJECT_SOURCE_DIR}/../cnn_layer_accel/model/inc/
    ${PROJECT_SOURCE_DIR}/../network/inc/
    ${PROJECT_SOURCE_DIR}/../myNetProto/inc/
)
# END Include directories ---------------------------------------------------------------------------------------------------------------------------


# BEGIN Library Settings ----------------------------------------------------------------------------------------------------------------------------
# Libraries to be linked against
#   For shared libraries make sure.....
#       WINDOWS: PATH env variable includes path(s) to dll's
#       LINUX:  LD_LIBRARY_PATH env variablel includes path(s) to so's
if(WIN32)
    TARGET_LINK_LIBRARIES(${TARGET_NAME} PUBLIC
        espresso.lib
    )
else()
    TARGET_LINK_LIBRARIES(${TARGET_NAME} PUBLIC
        -L/usr/local/lib
        -L/usr/lib/
        -L$ENV{WORKSPACE_PATH}/darknet/
		-L$ENV{WORKSPACE_PATH}/darknet/build/
        -L$ENV{IZO5011_X86_DEV_TOOLS}/lib/
		-L${PROJECT_SOURCE_DIR}/../util/build
		-L${PROJECT_SOURCE_DIR}/../fixedPoint/build/${BUILD_DIR_SUFFIX}/
		-L${PROJECT_SOURCE_DIR}/../espresso/build/${BUILD_DIR_SUFFIX}/
        -L${PROJECT_SOURCE_DIR}/../network/build/${BUILD_DIR_SUFFIX}/
        -L${PROJECT_SOURCE_DIR}/../myNetProto/build/${BUILD_DIR_SUFFIX}/
        -L${PROJECT_SOURCE_DIR}/../SYSC_FPGA_shim/build/${BUILD_DIR_SUFFIX}/
        -L${PROJECT_SOURCE_DIR}/../caffeDataParser/build/${BUILD_DIR_SUFFIX}/
		#-lcaffe
        #-lprotobuf
        #-lopencv_core
        #-lopencv_highgui
        #-lopencv_imgproc
		-ldarknet
		#caffeClassifier
        #caffeDataParser
		-lutil
        -lespresso
        -lsysc_fpga_shim
        -lfixedPoint
        -lnetwork
        -lmyNetProto
        -lcaffeDataParser
        -lprotobuf
        -lpthread
        #RL
)
endif()
# END Library Settings ------------------------------------------------------------------------------------------------------------------------------




