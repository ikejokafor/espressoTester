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
cmake_minimum_required (VERSION 3.10.1)
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


# BEGIN Set Output Path for libs or bins ------------------------------------------------------------------------------------------------------------
# Option 1) Executable
#    SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
#                            RUNTIME_OUTPUT_DIRECTORY 
#                            ${PROJECT_SOURCE_DIR}/<path>)
# Option 2) Static lib                           
#    SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
#                            LIBRARY_OUTPUT_DIRECTORY 
#                            ${PROJECT_SOURCE_DIR}/<path>)
# Option 3) Shared lib                           
#    SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
#                            ARCHIVE_OUTPUT_DIRECTORY 
#                            ${PROJECT_SOURCE_DIR}/<path>)                        
# END Set Output Path for libs or bins --------------------------------------------------------------------------------------------------------------


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
    ${PROJECT_SOURCE_DIR}/inc/*.h 
    ${PROJECT_SOURCE_DIR}/inc/*.hpp
    ${PROJECT_SOURCE_DIR}/inc/*.cuh
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


# BEGIN Build dependencies --------------------------------------------------------------------------------------------------------------------------
add_dependencies(${TARGET_NAME} espresso)
add_dependencies(${TARGET_NAME} caffeDataParser)
add_dependencies(${TARGET_NAME} fixedPoint)
# END Build dependencies ----------------------------------------------------------------------------------------------------------------------------


# BEGIN Debug / Release Settings --------------------------------------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKLE_CXX_FLAGS_DEBUG "-g")
set(CMAKLE_CXX_FLAGS_RELEASE "-O3")
# END Debug / Release Settings ----------------------------------------------------------------------------------------------------------------------


# BEGIN Compiler Flags ------------------------------------------------------------------------------------------------------------------------------
TARGET_COMPILE_OPTIONS(${TARGET_NAME} PUBLIC 
    "-std=c++11" 
    "-Wall"
    "-fexceptions"
)
# END Compiler Flags --------------------------------------------------------------------------------------------------------------------------------


# BEGIN Include directories -------------------------------------------------------------------------------------------------------------------------
TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/inc/
    ${PROJECT_SOURCE_DIR}/../caffeDataParser/inc/
    ${PROJECT_SOURCE_DIR}/../espresso/inc/
    ${PROJECT_SOURCE_DIR}/../fixedPoint/inc/
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
        -lprotobuf
        -lopencv_core
        -lopencv_highgui
        -lopencv_imgproc
        caffeDataParser
        espresso
        fixedPoint
)
endif()
# END Library Settings ------------------------------------------------------------------------------------------------------------------------------




