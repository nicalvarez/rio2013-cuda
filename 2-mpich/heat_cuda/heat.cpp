#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "colorstuff.h"
#include "sdlstuff.h"
#include "helpers_cuda.h"
#include "heat_cuda.h"


#define TAG 0
// grid size
static const unsigned int WIDTH = 256;
static const unsigned int HEIGHT = 256;
static const unsigned int LOCAL_HEIGHT = HEIGHT/2 + 1;

// heat source
static const float HEAT_TEMP = 5000.0f;
static const unsigned int HEAT_X = 130;
static const unsigned int HEAT_Y = 120;

// simulation parameters
static const unsigned int STEPS = 30000;


// calcular el indice lineal de un par de coordenadas
// en una matriz de ancho WIDTH
static size_t idx(unsigned int x, unsigned int y)
{
    return (size_t) y * WIDTH + x;
}

static float* row_pointer(float *a, int row) {
    return a + (row * WIDTH);
}

void swap_halo(int rank, int sender, int receiver, float* s, float* r) {
    MPI_Status status;
    if (rank == sender)
	MPI_Send(s, WIDTH, MPI_FLOAT, receiver, TAG, MPI_COMM_WORLD);
    else
	MPI_Recv(r, WIDTH, MPI_FLOAT, sender, TAG, MPI_COMM_WORLD, &status);
}

int main(int argc, char ** argv)
{
    int rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
    CHECK_CUDA_CALL(cudaSetDevice(rank));
    MPI_Init(&argc, &argv);

    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    assert(nproc == 2);
    fprintf(stderr, "proceso %d: iniciando\n", rank); 

    size_t local_grid_size = WIDTH * LOCAL_HEIGHT * sizeof(float),
	   grid_size = WIDTH * HEIGHT * sizeof(float);

    // pedir memoria
    fprintf(stderr, "proceso %d: alocando memoria...\n", rank);
    float * current;
    float * next;
    float * result;
    CHECK_CUDA_CALL(cudaMalloc(&current, local_grid_size));
    CHECK_CUDA_CALL(cudaMalloc(&next, local_grid_size));
    CHECK_CUDA_CALL(cudaMallocHost(&result, grid_size));
    fprintf(stderr, "proceso %d: memoria reservada con exito...\n", rank);   
    
    float *send, *recv;
    if (rank == 0) {
	send = row_pointer(next, LOCAL_HEIGHT-2);
	recv = row_pointer(next, LOCAL_HEIGHT-1);
    }
    else {
	recv = row_pointer(next, 0);
	send = row_pointer(next, 1);
    }

    int firstRow = (rank == 0 ? 0 : LOCAL_HEIGHT-2);

    // inicializar con la fuente de calor
    CHECK_CUDA_CALL(cudaMemset(current, 0, local_grid_size));
    size_t local_heat_y = HEAT_Y - firstRow;
    if (0 <= local_heat_y && local_heat_y < LOCAL_HEIGHT) {
	CHECK_CUDA_CALL(cudaMemcpy(&current[idx(HEAT_X, local_heat_y)], &HEAT_TEMP, sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_CALL(cudaMemcpy(next, current, local_grid_size, cudaMemcpyDeviceToDevice));

    // correr las actualizaciones
    for (unsigned int step = 0; step < STEPS; ++step) {
        update_cuda(WIDTH, 1, WIDTH-1, 1, LOCAL_HEIGHT-1, HEAT_X, HEAT_Y, current, next);
/*	swap_halo(rank, 0, 1, send, recv);
	swap_halo(rank, 1, 0, send, recv);
*/
	MPI_Status status;
	if (rank == 0) MPI_Send(&next[idx(0,LOCAL_HEIGHT-2)], WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD);
	else MPI_Recv(&next[idx(0,0)], WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD, &status);

	if (rank == 0) MPI_Recv(&next[idx(0,LOCAL_HEIGHT-1)], WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &status);
	else MPI_Send(&next[idx(0,1)], WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD);

        float * swap = current;
        current = next;
        next = swap;
    }
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // copiar el resultado al host para graficar
    CHECK_CUDA_CALL(cudaMemcpy(result, current, local_grid_size, cudaMemcpyDeviceToHost));
    if (rank == 1)
	MPI_Send(result, WIDTH*HEIGHT/2, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD);
    else {
	MPI_Status status;
	MPI_Recv(row_pointer(result, HEIGHT/2), WIDTH*HEIGHT/2, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &status);
    }

    // graficos
    if (rank == 0) {
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
	free(gfx);
    }

    CHECK_CUDA_CALL(cudaFree(current));
    CHECK_CUDA_CALL(cudaFree(next));
    CHECK_CUDA_CALL(cudaFreeHost(result));

    MPI_Finalize();

    return 0;
}
