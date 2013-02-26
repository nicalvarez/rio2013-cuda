#include "helpers_cuda.h"
#include "heat_cuda.h"


static const unsigned int BLOCK_WIDTH = 16;
static const unsigned int BLOCK_HEIGHT = 16;


// calcular el indice lineal de un par de coordenadas
// en una matriz de ancho WIDTH
static __host__ __device__ size_t idx(unsigned int x, unsigned int y, unsigned int stride)
{
    return (size_t) y * stride + x;
}


// kernel que actualiza valores en [from_x, to_x) X [from_y, to_y)
static __global__ void update(unsigned int row_stride,
                              unsigned int from_x, unsigned int to_x,
                              unsigned int from_y, unsigned int to_y,
                              unsigned int heat_x, unsigned int heat_y,
                              const float * current,
                              float * next)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + from_x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + from_y;

    if (y < to_y && x < to_x && (y != heat_y || x != heat_x)) {
        next[idx(x, y, row_stride)] = (current[idx(x, y - 1, row_stride)] +
                                       current[idx(x - 1, y, row_stride)] +
                                       current[idx(x + 1, y, row_stride)] +
                                       current[idx(x, y + 1, row_stride)]) * 0.25f;
    }
}


void update_cuda(unsigned int row_stride,
                 unsigned int from_x, unsigned int to_x,
                 unsigned int from_y, unsigned int to_y,
                 unsigned int heat_x, unsigned int heat_y,
                 const float * current,
                 float * next)
{
   // configurar la grilla para el kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(div_round_up(to_x - from_x, block.x), div_round_up(to_y - from_y, block.y));
    update<<<grid, block>>>(row_stride, from_x, to_x, from_y, to_y, heat_x, heat_y, current, next);
}
