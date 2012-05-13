/** 
Functions to add and subtract frames from each other.
Very basic implementations. Is it faster or slower to use shared memory here? I think it's much faster, especially if coalescing can
be used.
*/

__global__ void
subtract_s16_u8(int16_t* dst, int dstride, uint8_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_U8(src, __mul24(yy, sstride));

    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] -= src[offset];
}

__global__ void
subtract_s16_s16(int16_t* dst, int dstride, int16_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_S16(src, __mul24(yy, sstride));

    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] -= src[offset];
}

__global__ void
add_s16_u8(int16_t* dst, int dstride, uint8_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_U8(src, __mul24(yy, sstride));

    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] += src[offset];
}

__global__ void
add_s16_s16(int16_t* dst, int dstride, int16_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_S16(src, __mul24(yy, sstride));

    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] += src[offset];
}
