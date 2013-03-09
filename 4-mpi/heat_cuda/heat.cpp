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
    static float *h_buffer;
    if (!h_buffer)
	CHECK_CUDA_CALL(cudaMallocHost(&h_buffer, WIDTH * sizeof(float)));

    MPI_Status status;
    if (rank == sender) {
	CHECK_CUDA_CALL(cudaMemcpy(h_buffer, s, WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	MPI_Send(h_buffer, WIDTH, MPI_FLOAT, receiver, 0, MPI_COMM_WORLD);
	
    }
    else {
	MPI_Recv(h_buffer, WIDTH, MPI_FLOAT, sender, 0, MPI_COMM_WORLD, &status);
	CHECK_CUDA_CALL(cudaMemcpy(r, h_buffer, WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    }
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    assert(nproc == 2);

    CHECK_CUDA_CALL(cudaSetDevice(rank));
    fprintf(stderr, "proceso %d: iniciando\n", rank); 

    size_t local_grid_size = WIDTH * LOCAL_HEIGHT * sizeof(float),
	   grid_size = WIDTH * HEIGHT * sizeof(float);

    // pedir memoria
    fprintf(stderr, "proceso %d: alocando memoria...\n", rank);
    float * current;
    float * next;
    float * result;
    float * buffer;
    CHECK_CUDA_CALL(cudaMalloc(&current, local_grid_size));
    CHECK_CUDA_CALL(cudaMalloc(&next, local_grid_size));
    CHECK_CUDA_CALL(cudaMallocHost(&buffer, WIDTH * sizeof(float))); 
    CHECK_CUDA_CALL(cudaMallocHost(&result, grid_size));

    fprintf(stderr, "proceso %d: memoria reservada con exito...\n", rank);   
    
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

	MPI_Status status;
	if (rank == 0) {
	    CHECK_CUDA_CALL(cudaMemcpy(buffer, &next[idx(0,LOCAL_HEIGHT-2)], WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
	    MPI_Send(buffer, WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD);
	}
	else {
	    MPI_Recv(buffer, WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD, &status);
	    CHECK_CUDA_CALL(cudaMemcpy(&next[idx(0,0)], buffer, WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	}

	if (rank == 0) {
	    MPI_Recv(buffer, WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &status);
	    CHECK_CUDA_CALL(cudaMemcpy(&next[idx(0,LOCAL_HEIGHT-1)], buffer, WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	}
	else {
	    CHECK_CUDA_CALL(cudaMemcpy(buffer, &next[idx(0,1)], WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
	    MPI_Send(buffer, WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD);
	}
			

/*	MPI_Status status;
	if (rank == 0) MPI_Send(&next[idx(0,LOCAL_HEIGHT-2)], WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD);
	else MPI_Recv(&next[idx(0,0)], WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD, &status);

	if (rank == 0) MPI_Recv(&next[idx(0,LOCAL_HEIGHT-1)], WIDTH, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &status);
	else MPI_Send(&next[idx(0,1)], WIDTH, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD);
*/
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
