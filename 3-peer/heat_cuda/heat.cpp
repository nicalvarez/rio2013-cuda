#include <stdio.h>
#include <stdlib.h>

#include "colorstuff.h"
#include "sdlstuff.h"
#include "helpers_cuda.h"
#include "heat_cuda.h"


// grid size
static const unsigned int WIDTH = 256;
static const unsigned int HEIGHT = 256;
static const unsigned int LOCAL_HEIGHT = HEIGHT/2 + 1;

// heat source
static const float HEAT_TEMP = 5000.0f;
static const unsigned int HEAT_X = 130;
static const unsigned int HEAT_Y = 127;

// simulation parameters
static const unsigned int STEPS = 30000;


// calcular el indice lineal de un par de coordenadas
// en una matriz de ancho WIDTH
static size_t idx(unsigned int x, unsigned int y)
{
    return (size_t) y * WIDTH + x;
}

static float* row_pointer(float* a, int row) {
    return a + row*WIDTH; 
}

int main(int argc, char ** argv)
{
    size_t grid_size = WIDTH * HEIGHT * sizeof(float),
	   local_grid_size = WIDTH * LOCAL_HEIGHT * sizeof(float);

    float * result;
    CHECK_CUDA_CALL(cudaMallocHost(&result, grid_size));

    float * current[2];
    float * next[2];
    for (int dev = 0; dev < 2; dev++) {
	CHECK_CUDA_CALL(cudaSetDevice(dev));
	
	// pedir memoria
	CHECK_CUDA_CALL(cudaMalloc(&current[dev], local_grid_size));
	CHECK_CUDA_CALL(cudaMalloc(&next[dev], local_grid_size));

	// inicializar con la fuente de calor
	int firstRow = (dev == 0 ? 0 : LOCAL_HEIGHT-1);
	CHECK_CUDA_CALL(cudaMemset(current[dev], 0, local_grid_size));
	
	// CHEQUEAR FUENTE DE CALOR
	if (HEAT_Y - firstRow < LOCAL_HEIGHT)
	    CHECK_CUDA_CALL(cudaMemcpy(&current[dev][idx(HEAT_X, HEAT_Y - firstRow)], &HEAT_TEMP, sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_CALL(cudaMemcpy(next[dev], current[dev], local_grid_size, cudaMemcpyDeviceToDevice));
    }
    

    // correr las actualizaciones
    for (unsigned int step = 0; step < STEPS; ++step) {
	for (int dev = 0; dev < 2; dev++) {
	    CHECK_CUDA_CALL(cudaSetDevice(dev));
	    update_cuda(WIDTH, 1, WIDTH-1, 1, LOCAL_HEIGHT-1, HEAT_X, HEAT_Y, current[dev], next[dev]);

	    float * swap = current[dev];
	    current[dev] = next[dev];
	    next[dev] = swap;
	}


	// Intercambiar halos
	CHECK_CUDA_CALL(cudaDeviceSynchronize());
	CHECK_CUDA_CALL(cudaMemcpy(
		    row_pointer(current[1], 0), 
		    row_pointer(current[0], LOCAL_HEIGHT-2), 
		    WIDTH * sizeof(float), 
		    cudaMemcpyDeviceToDevice
		    ));

	CHECK_CUDA_CALL(cudaMemcpy(
		    row_pointer(current[0], LOCAL_HEIGHT-1), 
		    row_pointer(current[1], 1), 
		    WIDTH * sizeof(float), 
		    cudaMemcpyDeviceToDevice
		    ));
    }
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // copiar el resultado al host para graficar
    CHECK_CUDA_CALL(cudaMemcpy(result, current[0], grid_size/2, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(row_pointer(result, HEIGHT/2), row_pointer(current[1],1), grid_size/2, cudaMemcpyDeviceToHost));

    // graficos
    sdls_init(WIDTH, HEIGHT);
    rgba * gfx = (rgba *) calloc(WIDTH * HEIGHT, sizeof(rgba));
    for (unsigned int y = 0; y < HEIGHT; ++y) {
        for (unsigned int x = 0; x < WIDTH; ++x) {
            gfx[idx(x, y)] = color1(result[idx(x, y)] / HEAT_TEMP);
	    if (y == HEIGHT/2) gfx[idx(x,y)] = color1(1);
        }
    }

    sdls_blitrectangle_rgba(0, 0, WIDTH, HEIGHT, gfx);
    sdls_draw();

    printf("Presione ENTER para salir\n");
    getchar();
    sdls_cleanup();

    for (int dev = 0; dev < 2; dev++) {
	CHECK_CUDA_CALL(cudaFree(current[dev]));
	CHECK_CUDA_CALL(cudaFree(next[dev]));
    }
    CHECK_CUDA_CALL(cudaFreeHost(result));
    free(gfx);

    return 0;
}
