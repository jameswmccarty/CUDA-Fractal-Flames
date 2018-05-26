#Makefile for CUDA based fractal generator

CC = nvcc
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) -arch sm_50 -D CUDA
CFLAGS = -ansi -O2
DFLAGS = -ansi -O0 -g 
LFLAGS = -lm -ltiff

all: fractal

fractal: fractal_cuda.cu
	$(CC) $(NVCCFLAGS) --compiler-options='$(CFLAGS)' fractal_cuda.cu -o fractal $(LFLAGS)

debug: fractal_cuda.cu
	$(CC) $(NVCCFLAGS) --compiler-options='$(DFLAGS)' fractal_cuda.cu -o fractal $(LFLAGS)

clean:
	\rm fractal
