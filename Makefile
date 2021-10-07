#
# 'make depend' uses makedepend to automatically generate dependencies 
#			   (dependencies are added to end of Makefile)
# 'make'		build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# get current working directory name and absolute directory path
CWD = $(notdir $(shell pwd))
CWD_A = $(shell pwd)
BUILD_DIR_SUFFIX = debug

# define the C compiler to use
CC = g++


# define any compile-time flags
# C++ compiler flags (-g -O0 -O1 -O2 -O3 -Wall -std=c++14 -std=c++11 -fPIC -fexceptions)
CFLAGS = -Wall -std=c++1y -g -fPIC -fexceptions -DSYSTEMC


# define any directories containing header files other than /usr/include
#
INCLUDES =  -I/usr/include/ \
			-I/usr/local/include/ \
			-I./inc/ \
			-I$(CWD_A)/../network/inc/ \
			-I$(CWD_A)/../syscNetProto/inc/ \
			-I$(CWD_A)/../caffeDataParser/inc/ \
			-I$(CWD_A)/../espresso/inc/ \
			-I$(CWD_A)/../espresso/inc/CPU/ \
			-I$(CWD_A)/../espresso/inc/FPGA/ \
			-I$(CWD_A)/../espresso/inc/Darknet/ \
			-I$(CWD_A)/../fixedPoint/inc/ \
			-I$(CWD_A)/../caffeClassifier/inc/ \
			-I$(CWD_A)/../darknet/include/ \
			-I$(CWD_A)/../darknet/src/ \
			-I$(CWD_A)/../util/inc/ \
			-I$(CWD_A)/../FPGA_shim/inc/ \
			-I$(CWD_A)/../SYSC_FPGA_shim/inc/ \
			-I$(CWD_A)/../cnn_layer_accel/model/inc/ \
			-I$(CWD_A)/../network/inc/ \
			-I$(CWD_A)/../syscNetProto/inc/
			


# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS =	-L/usr/lib/ \
			-L/usr/local/lib/ \
			-L$(CWD_A)/../network/build/debug/ \
			-L$(CWD_A)/../syscNetProto/build/debug/ \
			-L$(WORKSPACE_PATH)/darknet/ \
			-L$(WORKSPACE_PATH)/darknet/build/ \
			-L$(IZO5011_X86_DEV_TOOLS)/lib/ \
			-L$(CWD_A)/../util/build/ \
			-L$(CWD_A)/../fixedPoint/build/$(BUILD_DIR_SUFFIX)/ \
			-L$(CWD_A)/../espresso/build/$(BUILD_DIR_SUFFIX)/ \
			-L$(CWD_A)/../network/build/$(BUILD_DIR_SUFFIX)/ \
			-L$(CWD_A)/../syscNetProto/build/$(BUILD_DIR_SUFFIX)/ \
			-L$(CWD_A)/../SYSC_FPGA_shim/build/$(BUILD_DIR_SUFFIX)/ \
			-L$(CWD_A)/../caffeDataParser/build/$(BUILD_DIR_SUFFIX)/
			


# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS =  -ldarknet \
		-lutil \
		-lespresso \
		-lSYSC_FPGA_shim \
		-lfixedPoint \
		-lnetwork \
		-lsyscNetProto \
		-lpthread \
		-lm
		# -lcaffeDataParser
		# -lprotobuf


# define the C source files
SRCS := $(wildcard ./src/*.cpp)
# $(info $(SRC))


# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#		 For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.cpp=.o)


# define the executable file 
TARGET = build/debug/$(CWD)


#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#
.PHONY: depend clean


default: $(TARGET)


$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)


# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


clean:
	$(RM) *.o *~ $(TARGET)


depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
