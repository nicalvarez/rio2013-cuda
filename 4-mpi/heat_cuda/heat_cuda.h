#pragma once

/* Realizar un paso de actualizacion en la GPU */
void update_cuda(unsigned int row_stride,
                 unsigned int from_x, unsigned int to_x,
                 unsigned int from_y, unsigned int to_y,
                 unsigned int heat_x, unsigned int heat_y,
                 const float * current,
                 float * next);
