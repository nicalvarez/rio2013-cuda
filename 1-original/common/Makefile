CXX=g++
CXXFLAGS=-O3 -fPIC -Wall -Wno-unused-function `sdl-config --cflags`

TARGETS=libsdlstuff.a

all: $(TARGETS)

libsdlstuff.a: sdlstuff.o
	ar rvs $@ $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
