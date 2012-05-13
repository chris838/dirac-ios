/** Clamp a value in 0..255 */
__device__ uint8_t u8_s16(int16_t i)
{
    return min(max(i+128, 0), 255);
}

__global__ void
convert_u8_s16(uint8_t* dst, int dstride, int dwidth, int16_t* src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    dst = OFFSET_U8(dst, __mul24(yy, dstride));
    src = OFFSET_S16(src, __mul24(min(yy, sheight-1), sstride));
    
    int minwidth = min(swidth, dwidth);
    /** Copy aligned part of line */
    int offset = tid*4;
    for(; offset < minwidth; offset += THREADS*4)
    {
        i16_4 s = *((i16_4*)&src[offset]);
        u8_4 d;
        d.a = u8_s16(s.a);
        d.b = u8_s16(s.b);
        d.c = u8_s16(s.c);
        d.d = u8_s16(s.d);
        *((u8_4*)&dst[offset]) = d;
    }

    /** Handle non-aligned part of line 
        minwidth&~3 .. (minwidth+3)&~3
    */
    uint8_t val = u8_s16(src[swidth - 1]);
    offset = minwidth&~3;
    u8_4 d;
    if(tid==0 && (minwidth&3))
    {
        i16_4 s = *((i16_4*)&src[offset]);
        d.d = val;
        d.a = u8_s16(s.a);
        if((minwidth&3) >= 2)
            d.b = u8_s16(s.b);
        else
            d.b = val;
        if((minwidth&3) >= 3)
            d.c = u8_s16(s.c);
        else
            d.c = val;
        *((u8_4*)&dst[offset]) = d;
    }

    /** Pad up the rest of destination line */
    offset = ((minwidth+3)&(~3)) + tid;
    dwidth = (dwidth+3)&(~3);
    d.a = d.b = d.c = d.d = val;
    
    for(; offset < dwidth; offset += THREADS*4)
        *((u8_4*)&dst[offset]) = d;
}

__device__ int16_t s16_u8(uint8_t i)
{
    return ((int16_t)i)-128;
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
        dst[offset] = s16_u8(src[offset]);

    /** Pad up the rest of destination line */
    int16_t val = s16_u8(src[swidth - 1]);
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
