#include <stdio.h>
#include <stdlib.h>

#include "colorstuff.h"
#include "sdlstuff.h"
#include "helpers_cuda.h"
#include "heat_cuda.h"


// grid size
static const unsigned int WIDTH = 256;
static const unsigned int HEIGHT = 256;

// heat source
static const float HEAT_TEMP = 5000.0f;
static const unsigned int HEAT_X = 30;
static const unsigned int HEAT_Y = 110;

// simulation parameters
static const unsigned int STEPS = 30000;


// calcular el indice lineal de un par de coordenadas
// en una matriz de ancho WIDTH
static size_t idx(unsigned int x, unsigned int y)
{
    return (size_t) y * WIDTH + x;
}


int main(int argc, char ** argv)
{
    size_t grid_size = WIDTH * HEIGHT * sizeof(float);

    // pedir memoria
    float * current;
    float * next;
    float * result;
    CHECK_CUDA_CALL(cudaMalloc(&current, grid_size));
    CHECK_CUDA_CALL(cudaMalloc(&next, grid_size));
    CHECK_CUDA_CALL(cudaMallocHost(&result, grid_size));

    // inicializar con la fuente de calor
    CHECK_CUDA_CALL(cudaMemset(current, 0, grid_size));
    CHECK_CUDA_CALL(cudaMemcpy(&current[idx(HEAT_X, HEAT_Y)], &HEAT_TEMP, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(next, current, grid_size, cudaMemcpyDeviceToDevice));

    // correr las actualizaciones
    for (unsigned int step = 0; step < STEPS; ++step) {
        update_cuda(WIDTH, 1, WIDTH-1, 1, HEIGHT-1, HEAT_X, HEAT_Y, current, next);

        float * swap = current;
        current = next;
        next = swap;
    }
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // copiar el resultado al host para graficar
    CHECK_CUDA_CALL(cudaMemcpy(result, current, grid_size, cudaMemcpyDeviceToHost));

    // graficos
    sdls_init(WIDTH, HEIGHT);
    rgba * gfx = (rgba *) calloc(WIDTH * HEIGHT, sizeof(rgba));
    for (unsigned int y = 0; y < HEIGHT; ++y) {
        for (unsigned int x = 0; x < WIDTH; ++x) {
            gfx[idx(x, y)] = color1(result[idx(x, y)] / HEAT_TEMP);
        }
    }

    sdls_blitrectangle_rgba(0, 0, WIDTH, HEIGHT, gfx);
    sdls_draw();

    printf("Presione ENTER para salir\n");
    getchar();
    sdls_cleanup();

    CHECK_CUDA_CALL(cudaFree(current));
    CHECK_CUDA_CALL(cudaFree(next));
    CHECK_CUDA_CALL(cudaFreeHost(result));
    free(gfx);

    return 0;
}
