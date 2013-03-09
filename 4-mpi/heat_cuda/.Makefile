CXX=/opt/mvapich2-1.9a2/bin/mpicxx
CPPFLAGS=-I/opt/cuda/include -I../common `sdl-config --cflags`
CXXFLAGS=
LDFLAGS=-L/opt/cuda/lib64 -L../common -lcudart -lm `sdl-config --libs` -lSDL_image -lsdlstuff

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

heat: heat.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
