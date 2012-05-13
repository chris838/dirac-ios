/// XXX shared memory, coalescing

/** Clamp a value in 0..255 */
__device__ uint8_t clamp_u8(int16_t i)
{
    return min(max(i, 0), 255);
}

__global__ void
convert_u8_s16(uint8_t* dst, int dstride, int dwidth, int16_t* src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_U8(dst, __mul24(yy, dstride));
    src = OFFSET_S16(src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] = clamp_u8(src[offset]);

    /** Pad up the rest of destination line */
    uint8_t val = clamp_u8(src[swidth - 1]);
    for(; offset < dwidth; offset += THREADS)
        dst[offset] = val;
}

__global__ void
convert_s16_u8(int16_t* dst, int dstride, int dwidth, uint8_t* src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_U8(src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] = src[offset];

    /** Pad up the rest of destination line */
    int16_t val = src[swidth - 1];
    for(; offset < dwidth; offset += THREADS)
        dst[offset] = val;
}

__global__ void
convert_u8_u8(uint8_t* dst, int dstride, int dwidth, uint8_t* src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_U8(dst, __mul24(yy, dstride));
    src = OFFSET_U8(src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
        dst[offset] = src[offset];

    /** Pad up the rest of destination line */
    uint8_t val = src[swidth - 1];
    for(; offset < dwidth; offset += THREADS)
        dst[offset] = val;
}

__global__ void
convert_s16_s16(int16_t* dst, int dstride, int dwidth, int16_t* src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_S16(dst, __mul24(yy, dstride));
    src = OFFSET_S16(src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
    {
        dst[offset] = src[offset];
    }

    /** Pad up the rest of destination line */
    int16_t val = src[swidth - 1];
    for(; offset < dwidth; offset += THREADS)
    {
        dst[offset] = val;
    }
}
