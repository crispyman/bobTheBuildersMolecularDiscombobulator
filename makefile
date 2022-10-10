NVCC = nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
# -fmad=false is reqired to get results to match between cpu and gpu with optimzation -use_fast_math
NVCCFLAGS = -c -O3 -fmad=false  -Xptxas -O3 -lineinfo --compiler-options  "-Wall -O3 -mavx2 -mfma"

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options "-g -Wall "

OBJS = wrappers.o main.o csvparser.o csvwriter.o h_main.o d_main.o
.SUFFIXES: .cu .o .h
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

main: $(OBJS) makefile
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o main

main.o: main.cu csvparser.h csvwriter.h CHECK.h wrappers.cu main.h h_main.h d_main.h molecule.h config.h makefile

csvparser.o: csvparser.cu csvparser.h CHECK.h makefile

csvwriter.o: csvwriter.cu csvwriter.h CHECK.h makefile

h_main.o: h_main.cu h_main.h CHECK.h config.h makefile

d_main.o: d_main.cu d_main.h CHECK.h config.h makefile

wrappers.o: wrappers.cu wrappers.h makefile

clean:
	rm main *.o
