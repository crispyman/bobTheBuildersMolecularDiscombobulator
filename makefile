NVCC = nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
NVCCFLAGS = -c -O2 -Xptxas -O2 -lineinfo --compiler-options -Wall

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = wrappers.o main.o csvparser.o h_main.o d_main.o
.SUFFIXES: .cu .o .h
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

main: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o main

main.o: main.cu csvparser.h CHECK.h wrappers.cu main.h h_main.h d_main.h config.h molecule.h

csvparser.o: csvparser.cu csvparser.h CHECK.h

h_main.o: h_main.cu h_main.h CHECK.h

d_main.o: d_main.cu d_main.h CHECK.h


wrappers.o: wrappers.cu wrappers.h

clean:
	rm main *.o
