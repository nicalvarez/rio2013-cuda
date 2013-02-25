#pragma once

#include <mpi.h>

// Helper for CHECK_MPI_CALL
static inline void _checkMPIReturnValue(int result,
    char const * const func,
    const char * const file,
    int const line) 
{
    if (result != MPI_SUCCESS) {
        int error_len = 0;
        char error_msg[MPI_MAX_ERROR_STRING];
        
        MPI_Error_string(result, error_msg, &error_len);
        fprintf(stderr, "MPI error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, result, error_msg, func);
        MPI_Abort(MPI_COMM_WORLD, result);
    }
}


// Check the result of a MPI call and abort with information on error
#define CHECK_MPI_CALL(f) _checkMPIReturnValue( (f), #f, __FILE__, __LINE__ )
