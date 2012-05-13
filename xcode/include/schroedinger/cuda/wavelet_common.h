#define DATATYPE int16_t

#define HORIZONTAL
#define VERTICAL

// Horizontally use 256 threads
#define BSH 256

/// Vertically use a grid of 16*8 threads
#define BCOLS_SHIFT 5
#define BCOLS 32 /* Columns to process at once, must be 2 times number of threads in X dir */

#define BSVX 16 /* Width of thread matrix */
#define BSVY 8 /* Height of thread matrix */
