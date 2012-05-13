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

    i16_4* _dst = (i16_4*)OFFSET_S16(dst, __mul24(yy, dstride));
    u8_4* _src = (u8_4*)OFFSET_U8(src, __mul24(yy, sstride));

    int offset = tid;
    minwidth = (minwidth+3)>>2;
    for(; offset < minwidth; offset += THREADS)
    {
        u8_4 s = _src[offset];
        i16_4 d = _dst[offset];
        d.a -= s.a;
        d.b -= s.b;
        d.c -= s.c;
        d.d -= s.d;
        _dst[offset] = d;
    }
}

__global__ void
subtract_s16_s16(int16_t* dst, int dstride, int16_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    i16_2* _dst = (i16_2*)OFFSET_S16(dst, __mul24(yy, dstride));
    i16_2* _src = (i16_2*)OFFSET_S16(src, __mul24(yy, sstride));

    int offset = tid;
    minwidth = (minwidth+1)>>1; // round up to 2
    for(; offset < minwidth; offset += THREADS)
    {
        i16_2 s = _src[offset];
        i16_2 d = _dst[offset];
        d.a -= s.a;
        d.b -= s.b;
        _dst[offset] = d;
    }
}

__global__ void
add_s16_u8(int16_t* dst, int dstride, uint8_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    i16_4* _dst = (i16_4*)OFFSET_S16(dst, __mul24(yy, dstride));
    u8_4* _src = (u8_4*)OFFSET_U8(src, __mul24(yy, sstride));

    int offset = tid;
    minwidth = (minwidth+3)>>2;
    for(; offset < minwidth; offset += THREADS)
    {
        u8_4 s = _src[offset];
        i16_4 d = _dst[offset];
        d.a += s.a;
        d.b += s.b;
        d.c += s.c;
        d.d += s.d;
        _dst[offset] = d;
    }
}

__global__ void
add_s16_s16(int16_t* dst, int dstride, int16_t* src, int sstride, int minwidth)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    i16_2* _dst = (i16_2*)OFFSET_S16(dst, __mul24(yy, dstride));
    i16_2* _src = (i16_2*)OFFSET_S16(src, __mul24(yy, sstride));

    int offset = tid;
    minwidth = (minwidth+1)>>1; // round up to 2
    for(; offset < minwidth; offset += THREADS)
    {
        i16_2 s = _src[offset];
        i16_2 d = _dst[offset];
        d.a += s.a;
        d.b += s.b;
        _dst[offset] = d;
    }
}
