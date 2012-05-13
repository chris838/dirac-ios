/// strides are in bytes

// dwidth should be half of actual width
// swidth in units of 4 bytes
__global__ void
convert_u8_422_yuyv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, uint8_t* _src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    dsty = OFFSET_U8(dsty, __mul24(yy, ystride));
    dstu = OFFSET_U8(dstu, __mul24(yy, ustride));
    dstv = OFFSET_U8(dstv, __mul24(yy, vstride));

    /** Clamp at image bottom using max */
    uint32_t *src = OFFSET_U32(_src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
    {
        uint32_t pixel = src[offset];

        *OFFSET_U16(dsty, offset<<1) = (pixel&0xFF)|((pixel>>8)&0xFF00);
        dstu[offset]       = pixel>>8;
        dstv[offset]       = pixel>>24;
    }

    /** Pad up the rest of destination line */
    uint32_t pixel = src[swidth - 1];
    uint16_t y = ((pixel>>16)&0xFF)|((pixel>>8)&0xFF00);
    uint8_t u = pixel>>8;
    uint8_t v = pixel>>24;
    for(; offset < dwidth; offset += THREADS)
    {
        *OFFSET_U16(dsty, offset<<1) = y;
        dstu[offset] = u;
        dstv[offset] = v;
    }
}

__global__ void
convert_u8_422_uyvy(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, uint8_t* _src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    dsty = OFFSET_U8(dsty, __mul24(yy, ystride));
    dstu = OFFSET_U8(dstu, __mul24(yy, ustride));
    dstv = OFFSET_U8(dstv, __mul24(yy, vstride));

    /** Clamp at image bottom using max */
    uint32_t *src = OFFSET_U32(_src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
    {
        uint32_t pixel = src[offset];

        *OFFSET_U16(dsty, offset<<1) = ((pixel>>8)&0xFF)|((pixel>>16)&0xFF00);
        dstu[offset]       = pixel;
        dstv[offset]       = pixel>>16;
    }

    /** Pad up the rest of destination line */
    uint32_t pixel = src[swidth - 1];
    uint16_t y = ((pixel>>24)&0xFF)|((pixel>>16)&0xFF00);
    uint8_t u = pixel;
    uint8_t v = pixel>>16;
    for(; offset < dwidth; offset += THREADS)
    {
        *OFFSET_U16(dsty, offset<<1) = y;
        dstu[offset] = u;
        dstv[offset] = v;
    }
}

/// unpack AYUV to Y,U,V; alpha component is ignored for now
/// swidth and dwidth is in pixels
__global__ void
convert_u8_444_ayuv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, uint8_t* _src, int sstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    dsty = OFFSET_U8(dsty, __mul24(yy, ystride));
    dstu = OFFSET_U8(dstu, __mul24(yy, ustride));
    dstv = OFFSET_U8(dstv, __mul24(yy, vstride));

    /** Clamp at image bottom using max */
    uint32_t *src = OFFSET_U32(_src, __mul24(min(yy, sheight-1), sstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
    {
        uint32_t pixel = src[offset];

        dstv[offset] = pixel>>24;
        dstu[offset] = pixel>>16;
        dsty[offset] = pixel>>8;
    }

    /** Pad up the rest of destination line */
    uint32_t pixel = src[swidth - 1];
    uint8_t v = pixel>>24;
    uint8_t u = pixel>>16;
    uint8_t y = pixel>>8;
    for(; offset < dwidth; offset += THREADS)
    {
        dsty[offset] = y;
        dstu[offset] = u;
        dstv[offset] = v;
    }
}


/// interleave/pack
// swidth should be half of actual width
// dwidth in units of 4 bytes
__global__ void
convert_yuyv_u8_422 (uint8_t* _dst, int dstride, int dwidth, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    uint32_t *dst = OFFSET_U32(_dst, __mul24(yy, dstride));

    /** Source row offsets */
    const int clampy = min(yy, sheight-1);
    srcy = OFFSET_U8(srcy, __mul24(clampy, ystride));
    srcu = OFFSET_U8(srcu, __mul24(clampy, ustride));
    srcv = OFFSET_U8(srcv, __mul24(clampy, vstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    uint32_t pixel;
    for(; offset < minwidth; offset += THREADS)
    {
        uint16_t y = *OFFSET_U16(srcy, offset<<1);
        pixel = (y&0xFF) | (srcu[offset]<<8) | ((y&0xFF00)<<8) | (srcv[offset]<<24);
        dst[offset] = pixel;
    }

    /** Pad up the rest of destination line */
    uint8_t y = srcy[(2*swidth)-1];
    pixel = y | (srcu[swidth-1]<<8) | (y<<16) | (srcv[swidth-1]<<24);
    for(; offset < dwidth; offset += THREADS)
    {
        dst[offset] = pixel;
    }
}

__global__ void
convert_uyvy_u8_422 (uint8_t* _dst, int dstride, int dwidth, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    uint32_t *dst = OFFSET_U32(_dst, __mul24(yy, dstride));

    /** Source row offsets */
    const int clampy = min(yy, sheight-1);
    srcy = OFFSET_U8(srcy, __mul24(clampy, ystride));
    srcu = OFFSET_U8(srcu, __mul24(clampy, ustride));
    srcv = OFFSET_U8(srcv, __mul24(clampy, vstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    uint32_t pixel;
    for(; offset < minwidth; offset += THREADS)
    {
        uint16_t y = *OFFSET_U16(srcy, offset<<1);
        pixel = (srcu[offset]) | ((y&0xFF)<<8) | (srcv[offset]<<16) | ((y&0xFF00)<<16);
        dst[offset] = pixel;
    }

    /** Pad up the rest of destination line */
    uint8_t y = srcy[(2*swidth)-1];
    pixel = srcu[swidth-1] | (y<<8) | (srcv[swidth-1]<<16) | (y<<24);
    for(; offset < dwidth; offset += THREADS)
    {
        dst[offset] = pixel;
    }
}

/// pack Y,U,V to AYUV; alpha component is ignored for now
/// swidth and dwidth is in pixels
__global__ void
convert_ayuv_u8_444 (uint8_t* _dst, int dstride, int dwidth, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    const int yy = blockIdx.x;     /* row */
    const int tid = threadIdx.x;   /* column offset */

    /** Destination row offset */
    uint32_t *dst = OFFSET_U32(_dst, __mul24(yy, dstride));

    /** Source row offsets */
    const int clampy = min(yy, sheight-1);
    srcy = OFFSET_U8(srcy, __mul24(clampy, ystride));
    srcu = OFFSET_U8(srcu, __mul24(clampy, ustride));
    srcv = OFFSET_U8(srcv, __mul24(clampy, vstride));

    /** Copy part of line */
    int minwidth = min(swidth, dwidth);
    int offset = tid;
    for(; offset < minwidth; offset += THREADS)
    {
        dst[offset] = (srcv[offset]<<24)|(srcu[offset]<<16)|(srcy[offset]<<8)|0xFF;
    }

    /** Pad up the rest of destination line */
    uint32_t pixel = (srcv[swidth-1]<<24)|(srcu[swidth-1]<<16)|(srcy[swidth-1]<<8)|0xFF;
    for(; offset < dwidth; offset += THREADS)
        dst[offset] = pixel;
}
