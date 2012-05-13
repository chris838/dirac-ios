// declare texture references
typedef texture<float, 2, cudaReadModeElementType> RefTex;
typedef texture<short4, 2, cudaReadModeElementType> BlockTex;

typedef int worktype;
typedef int2 worktype2;

BlockTex bt1;
RefTex ref1;
RefTex ref2;

#define PDIFF 8 /* Spacing between pixels in texture */
#define PDIFF_L2 3 /* Log2 of spacing between pixels in texture */
#define POFS 2  /* Texel offset */
#define SXSCALE 0.25f
#define SYSCALE 0.25f

__device__ worktype tex_u8(RefTex tex, short x, short y, short xx, short yy, short xshift, short yshift)
{
    return (worktype)(tex2D(tex, (x+(xx>>xshift))*SXSCALE, (y+(yy>>yshift))*SYSCALE)*256.0f);
}

struct __align__((4)) int16_2 { // 32 bit
    int16_t x,y;
};


__device__ worktype2 do_motion_block(short bidx, short bidy, short xofs, short yofs, short comp_shift,
  short weight1, short weight2, short weight_shift, short xshift, short yshift)
{
    /// Retrieve motion vector
    short4 mv = tex2D(bt1, (float)bidx, (float)bidy);

    worktype2 val;

    if(mv.x != MOTION_NONE)
    {
        if(mv.y != MOTION_NONE) // Use both references
        {
            worktype ref1_v,ref2_v;
            
            ref1_v = tex_u8(ref1, xofs, yofs, mv.x, mv.z, xshift, yshift);
            ref2_v = tex_u8(ref2, xofs, yofs, mv.y, mv.w, xshift, yshift);
            val.x = (__mul24(ref1_v, weight1) +
                     __mul24(ref2_v, weight2)) >> weight_shift;
            
            ref1_v = tex_u8(ref1, xofs + PDIFF, yofs, mv.x, mv.z, xshift, yshift);
            ref2_v = tex_u8(ref2, xofs + PDIFF, yofs, mv.y, mv.w, xshift, yshift);
            val.y = (__mul24(ref1_v, weight1) +
                     __mul24(ref2_v, weight2)) >> weight_shift;
        }
        else // Use only reference 1
        {
            val.x = tex_u8(ref1, xofs, yofs, mv.x, mv.z, xshift, yshift);
            val.y = tex_u8(ref1, xofs + PDIFF, yofs, mv.x, mv.z, xshift, yshift);
        }
    }
    else
    {
        if(mv.y != MOTION_NONE) // Use only reference 2
        {
            val.x = tex_u8(ref2, xofs, yofs, mv.y, mv.w, xshift, yshift);
            val.y = tex_u8(ref2, xofs + PDIFF, yofs, mv.y, mv.w, xshift, yshift);
        }
        else // Only DC
        {
            /* As we can't index into registers, use bit shifting on the y1/y2 part */
            int yuv = (((uint16_t)mv.w)<<16)|((uint16_t)mv.z);
            val.x = val.y = ((yuv>>comp_shift)&0xFF);
        }
    }
    
    return val;
}

/* Divide image into blocks depending on xsep, xlen, ysep, ylen 

      Overlap     Origin                  Size
   -------------------------------------------------
   1: Inner       x*xsep,y*ysep           xmid,ymid
   2: Horizontal  x*xsep+xmid,y*ysep      xramp,ymid
   3: Vertical    x*xsep,y*ysep+ymid      xmid,yramp
   4: Diagonal    x*xsep+xmid,y*ysep+ymid xramp,yramp
   
   xramp = xlen-xsep
   xmid  = 2*xsep-xlen
   
   xramp_log2
   yramp_log2
   xsep_log2
   ysep_log2
   xmid_log2
   ymid_log2
   
   Thread matrix size is THREADX/THREADSY
       x per block                   y per block
   -----------------------------------------------------
   1:  THREADSX_LOG2-xmid_log2       THREADSY_LOG2-ymid_log2
   2:  ...
   
   
*/
/** The grid is divided into multiple areas, each area does a type of blending.
              xB
    +---------+---+
    |         |   |
    |    1    | 2 |
    |         |   |
 yB +---------+---+
    |    3    | 4 |
    +---------+---+
 */

/** Memory set-up */
#define STUFF(bx, by, xs_log2, ys_log2, xadd, yadd, xbadd, ybadd) \
    /** (Log2 of) number of motion blocks in x and y direction handled by this cuda block */ \
    short nx = THREADSX_LOG2 - (xs_log2); \
    short ny = THREADSY_LOG2 - (ys_log2); \
    /** Motion block offset for this thread */ \
    short obx = ((bx) << nx) + (threadIdx.x >> (xs_log2)) + xbadd; \
    short oby = ((by) << ny) + (threadIdx.y >> (ys_log2)) + ybadd; \
    /** Block id base for this thread */ \
    /* int blockid = __mul24(oby, obmc.blocksx) + obx;*/ \
    /** Offsets inside motion block */ \
    short ix = (threadIdx.x & ((1<<(xs_log2))-1)) << 1; \
    short iy = threadIdx.y & ((1<<(ys_log2))-1); \
    /** Offsets in image */ \
    short xofs = (obx<<xsep_log2) + ix + (xadd) + (1<<(xramp_log2-1)); \
    short yofs = (oby<<ysep_log2) + iy + (yadd) + (1<<(yramp_log2-1)); \
    /* Destination offset in memory */ \
    dest = OFFSET_S16(dest, __mul24(dstride, yofs) + (xofs<<1))

/* each thread calculates two pixels, write at once (must be aligned) */
__global__ void motion_copy_2ref_4b(
    int16_t* dest, int dstride, int dwidth, int dheight, int xB, int yB,
    // Component parameters
    short comp /* times 8 */,
    // Motion block parameters (depend on component too)
    int xramp_log2, int yramp_log2, int xsep_log2, int ysep_log2, int xmid_log2, int ymid_log2,
    // Blend weights
    short weight1, short weight2, short weight_shift,
    // Component
    short xshift, short yshift
)
{
    if(blockIdx.x < xB) 
    {
        if(blockIdx.y < yB) // 1: Inner (no blend)
        {
            STUFF(blockIdx.x, blockIdx.y, xmid_log2-1, ymid_log2, 0, 0, 0, 0);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                
                val = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                
                /* Write value to memory */
                int16_2 sval;
                sval.x = val.x - 128;
                sval.y = val.y - 128;
                *((int16_2*)dest) = sval;
            }
        }
        else  // 3: Vertical (blend two blocks)
        {
            STUFF(blockIdx.x, blockIdx.y - yB, xmid_log2-1, yramp_log2, 0, 1<<ymid_log2, 0, -1);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                worktype yw2 = min(iy+1, (1<<yramp_log2));
                worktype yw1 = (1<<yramp_log2) - yw2;
                
                worktype2 valy1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                worktype2 valy2 = do_motion_block(obx, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                
                val.x = (__mul24(yw1, valy1.x) + __mul24(yw2, valy2.x)) >> yramp_log2;
                val.y = (__mul24(yw1, valy1.y) + __mul24(yw2, valy2.y)) >> yramp_log2;
                      
                /* Write value to memory */
                int16_2 sval;
                sval.x = val.x - 128;
                sval.y = val.y - 128;
                *((int16_2*)dest) = sval;
            }
        }
    }
    else
    {
        if(blockIdx.y < yB) // 2: Horizontal (blend 2 blocks)
        {
            STUFF(blockIdx.x - xB, blockIdx.y, xramp_log2-1, ymid_log2, 1<<xmid_log2, 0, -1, 0);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                worktype xw2 = min(ix+1, (1<<xramp_log2));
                worktype xw1 = (1<<xramp_log2) - xw2;
                
                worktype2 valx1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                worktype2 valx2 = do_motion_block(obx+1, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                
                val.x = (__mul24(xw1, valx1.x) + __mul24(xw2, valx2.x)) >> xramp_log2;
                val.y = (__mul24(xw1, valx1.y) + __mul24(xw2, valx2.y)) >> xramp_log2;
                      
                /* Write value to memory */
                int16_2 sval;
                sval.x = val.x - 128;
                sval.y = val.y - 128;
                *((int16_2*)dest) = sval;
            }
        }
        else // Diagonal (blend 4 blocks)
        {
            STUFF(blockIdx.x - xB, blockIdx.y - yB, xramp_log2-1, yramp_log2, 1<<xmid_log2, 1<<ymid_log2, -1, -1);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                worktype2 valy1,valy2;
                worktype xw2 = min(ix+1, (1<<xramp_log2));
                worktype xw1 = (1<<xramp_log2) - xw2;
                
                {
                    worktype2 valx1y1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    worktype2 valx2y1 = do_motion_block(obx+1, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    valy1.x = __mul24(xw1, valx1y1.x) + __mul24(xw2, valx2y1.x);
                    valy1.y = __mul24(xw1, valx1y1.y) + __mul24(xw2, valx2y1.y);
                }
                {
                    worktype2 valx1y2 = do_motion_block(obx, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    worktype2 valx2y2 = do_motion_block(obx+1, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    valy2.x = __mul24(xw1, valx1y2.x) + __mul24(xw2, valx2y2.x);
                    valy2.y = __mul24(xw1, valx1y2.y) + __mul24(xw2, valx2y2.y);
                }

                worktype2 val;
                int yw2 = min(iy+1, (1<<yramp_log2));
                int yw1 = (1<<yramp_log2) - yw2;

                val.x = (__mul24(yw1, valy1.x) + __mul24(yw2, valy2.x)) >> (xramp_log2 + yramp_log2);
                val.y = (__mul24(yw1, valy1.y) + __mul24(yw2, valy2.y)) >> (xramp_log2 + yramp_log2);
                      
                /* Write value to memory */
                int16_2 sval;
                sval.x = val.x - 128;
                sval.y = val.y - 128;
                *((int16_2*)dest) = sval;
            }
        
        }
    }
}

/* each thread calculates two pixels, write on at a time */
__global__ void motion_copy_2ref_2b(
    int16_t* dest, int dstride, int dwidth, int dheight, int xB, int yB,
    // Component parameters
    short comp /* times 8 */,
    // Motion block parameters (depend on component too)
    int xramp_log2, int yramp_log2, int xsep_log2, int ysep_log2, int xmid_log2, int ymid_log2,
    // Blend weights
    short weight1, short weight2, short weight_shift,
    // Component
    short xshift, short yshift
)
{
    if(blockIdx.x < xB) 
    {
        if(blockIdx.y < yB) // 1: Inner (no blend)
        {
            STUFF(blockIdx.x, blockIdx.y, xmid_log2-1, ymid_log2, 0, 0, 0, 0);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                
                val = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                dest[0] = val.x - 128;
                dest[1] = val.y - 128;
            }
        }
        else  // 3: Vertical (blend two blocks)
        {
            STUFF(blockIdx.x, blockIdx.y - yB, xmid_log2-1, yramp_log2, 0, 1<<ymid_log2, 0, -1);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                worktype yw2 = min(iy+1, (1<<yramp_log2));
                worktype yw1 = (1<<yramp_log2) - yw2;
                
                worktype2 valy1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                worktype2 valy2 = do_motion_block(obx, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                
                dest[0] = ((__mul24(yw1, valy1.x) + __mul24(yw2, valy2.x)) >> yramp_log2) - 128;
                dest[1] = ((__mul24(yw1, valy1.y) + __mul24(yw2, valy2.y)) >> yramp_log2) - 128;
            }
        }
    }
    else
    {
        if(blockIdx.y < yB) // 2: Horizontal (blend 2 blocks)
        {
            STUFF(blockIdx.x - xB, blockIdx.y, xramp_log2-1, ymid_log2, 1<<xmid_log2, 0, -1, 0);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                worktype2 val;
                worktype xw2 = min(ix+1, (1<<xramp_log2));
                worktype xw1 = (1<<xramp_log2) - xw2;
                
                worktype2 valx1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                worktype2 valx2 = do_motion_block(obx+1, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                
                dest[0] = ((__mul24(xw1, valx1.x) + __mul24(xw2, valx2.x)) >> xramp_log2) - 128;
                dest[1] = ((__mul24(xw1, valx1.y) + __mul24(xw2, valx2.y)) >> xramp_log2) - 128;
            }
        }
        else // Diagonal (blend 4 blocks)
        {
            STUFF(blockIdx.x - xB, blockIdx.y - yB, xramp_log2-1, yramp_log2, 1<<xmid_log2, 1<<ymid_log2, -1, -1);
            if(xofs >= 0 && yofs >= 0 && xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                worktype2 valy1,valy2;
                worktype xw2 = min(ix+1, (1<<xramp_log2));
                worktype xw1 = (1<<xramp_log2) - xw2;
                
                {
                    worktype2 valx1y1 = do_motion_block(obx, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    worktype2 valx2y1 = do_motion_block(obx+1, oby, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    valy1.x = __mul24(xw1, valx1y1.x) + __mul24(xw2, valx2y1.x);
                    valy1.y = __mul24(xw1, valx1y1.y) + __mul24(xw2, valx2y1.y);
                }
                {
                    worktype2 valx1y2 = do_motion_block(obx, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    worktype2 valx2y2 = do_motion_block(obx+1, oby+1, (xofs<<PDIFF_L2) + POFS, (yofs<<PDIFF_L2) + POFS, comp, weight1, weight2, weight_shift, xshift, yshift);
                    valy2.x = __mul24(xw1, valx1y2.x) + __mul24(xw2, valx2y2.x);
                    valy2.y = __mul24(xw1, valx1y2.y) + __mul24(xw2, valx2y2.y);
                }

                int yw2 = min(iy+1, (1<<yramp_log2));
                int yw1 = (1<<yramp_log2) - yw2;

                dest[0] = ((__mul24(yw1, valy1.x) + __mul24(yw2, valy2.x)) >> (xramp_log2 + yramp_log2)) - 128;
                dest[1] = ((__mul24(yw1, valy1.y) + __mul24(yw2, valy2.y)) >> (xramp_log2 + yramp_log2)) - 128;
            }
        
        }
    }
}
