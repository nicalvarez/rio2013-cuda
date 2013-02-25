CXX=g++
CXXFLAGS=-g -ggdb -Wall -Wno-unused-function
CPPFLAGS=-I../common -I/opt/cuda/include

CU=nvcc
CUFLAGS=-g -G -arch=sm_20 -use_fast_math `sdl-config --cflags`

LDFLAGS=-L/opt/cuda/lib64 -lcudart -lm -L../common -lsdlstuff `sdl-config --libs` -lSDL_image

TARGETS=heat

all: $(TARGETS)

%.o: %.cu
	$(CU) $(CPPFLAGS) $(CUFLAGS) -o $@ -c $<

heat: heat.o heat_cuda.o
	make -C ../common
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
