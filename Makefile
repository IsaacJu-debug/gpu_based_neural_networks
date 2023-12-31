OBJDIR=obj
DEPDIR=dep
CUDA_INC=./cuda_helper

# Compilers
CC=mpic++
CUD=nvcc

TARGET := main

CPPSRCS := $(wildcard *.cpp)
CPPOBJS := $(CPPSRCS:%.cpp=$(OBJDIR)/%.o)

CUDSRCS := $(wildcard *.cu)
CUDOBJS := $(CUDSRCS:%.cu=$(OBJDIR)/%.o)

UTLSRCS := $(wildcard utils/*.cpp) 
UTLOBJS := $(UTLSRCS:utils/%.cpp=$(OBJDIR)/%.o)

OBJS := $(CPPOBJS) $(CUDOBJS) $(UTLOBJS)

DEPFILES := $(OBJS:$(OBJDIR)/%.o=$(DEPDIR)/%.d)

# Flags
# Armadillo include path
ARMADILLO_INC_PATH=/usr/workspace/ju1/03_CS_foundation_repos/01_CME/TPL/armadillo-12.6.7/include
CUDA_INC_PATH=/usr/tce/packages/cuda/cuda-10.1.168/include
# Flags
CFLAGS=-O2 -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_DONT_USE_WRAPPER
CUDFLAGS=-lineinfo -O2 -c -arch=compute_70 -code=sm_70 -Xcompiler -Wall

# Include flags (adding Armadillo include path)
INCFLAGS=-I$(CUDA_INC) -I$(ARMADILLO_INC_PATH) -I$(CUDA_INC_PATH)

# LDFLAGS=-lblas -llapack -lcublas -lcudart

LIB_PATH=/usr/lib64
# Define the Armadillo library path
ARMADILLO_LIB_PATH=/usr/workspace/ju1/03_CS_foundation_repos/01_CME/TPL/armadillo-12.6.7/lib64
LDFLAGS=-Wl,-Bstatic $(LIB_PATH)/libblas.a $(LIB_PATH)/liblapack.a -Wl,-Bdynamic -lgfortran -lcublas -lcudart -L$(ARMADILLO_LIB_PATH) -larmadillo

DEPFLAGS=-MT $@ -MMD -MF $(addprefix $(DEPDIR)/, $(notdir $*)).d

CC_CMD=$(CC) $(CFLAGS) $(INCFLAGS)
CU_CMD=$(CUD) $(CUDFLAGS) $(INCFLAGS)

# --fmad=false

$(TARGET): $(OBJS)
	$(CC)  -o $@ $(OBJS) $(LDFLAGS)

$(CPPOBJS): $(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(UTLOBJS): $(OBJDIR)/%.o: utils/%.cpp $(DEPDIR)/%.d 
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(CUDOBJS): $(OBJDIR)/%.o: %.cu $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CU_CMD) -c $< -o $@

$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -rf $(OBJDIR)/*.o $(DEPDIR)/*.d main

clean_slurm:
	rm -f slurm*

clean_all: clean
	rm -f Outputs/*