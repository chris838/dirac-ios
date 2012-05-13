

//struct __align__((8)) _MotionVectorC {
//    int32_t x2x1;
//    int32_t y2y1;
//};
//typedef unsigned long long _MotionVectorC;

/** XXX chroma version of obmc and global_motion */
__constant__ _Obmc obmc;
__constant__ _MotionVector motion_vectors[MOTION_MAX_BLOCKS];

// declare texture references
typedef texture<float, 2, cudaReadModeNormalizedFloat> RefTex;

RefTex ref1;
RefTex ref2;

__device__ int tex_u8(RefTex tex, float x, float y)
{
    return (int)(tex2D(tex, x, y)*256.0f);
}

struct __align__((4)) int16_2 { // 32 bit
    int16_t x,y;
};

#define PDIFF 2.0f /* Spacing between pixels in texture */
#define POFS 0.5f  /* Texel offset */
__device__ int2 do_motion_block(int blockid, float xofs, float yofs, int comp_shift, float sxscale, float syscale)
{
    /// Retrieve motion vector
    _MotionVector mv = motion_vectors[blockid];

    int2 val;

    if(mv.x1 != MOTION_NONE)
    {
        if(mv.x2 != MOTION_NONE) // Use both references
        {
            int ref1_v,ref2_v;
            
            ref1_v = tex_u8(ref1, xofs + mv.x1 * sxscale, yofs + mv.y1 * syscale);
            ref2_v = tex_u8(ref2, xofs + mv.x2 * sxscale, yofs + mv.y2 * syscale);
            val.x = (__mul24(ref1_v, obmc.weight1) +
                     __mul24(ref2_v, obmc.weight2)) >> obmc.weight_shift;
            
            ref1_v = tex_u8(ref1, xofs + mv.x1 * sxscale + PDIFF, yofs + mv.y1 * syscale);
            ref2_v = tex_u8(ref2, xofs + mv.x2 * sxscale + PDIFF, yofs + mv.y2 * syscale);
            val.y = (__mul24(ref1_v, obmc.weight1) +
                     __mul24(ref2_v, obmc.weight2)) >> obmc.weight_shift;
        }
        else // Use only reference 1
        {
            val.x = tex_u8(ref1, xofs + mv.x1 * sxscale, yofs + mv.y1 * syscale);
            val.y = tex_u8(ref1, xofs + mv.x1 * sxscale + PDIFF, yofs + mv.y1 * syscale);
        }
    }
    else
    {
        if(mv.x2 != MOTION_NONE) // Use only reference 2
        {
            val.x = tex_u8(ref2, xofs + mv.x2 * sxscale, yofs + mv.y2 * syscale);
            val.y = tex_u8(ref2, xofs + mv.x2 * sxscale + PDIFF, yofs + mv.y2 * syscale);
        }
        else // Only DC
        {
            /* As we can't index into registers, use bit shifting on the y1/y2 part */
            int yuv = (((uint16_t)mv.y2)<<16)|((uint16_t)mv.y1);
            val.x = val.y = ((yuv>>comp_shift)&0xFF);
        }
    }
    
    return val;
}

// THREADSX_LOG2,  THREADSY_LOG2
// THREADX,        THREADSY

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

/** Memory set-up */
#define STUFF(bx, by, xs_log2, ys_log2, xadd, yadd) \
    /** (Log2 of) number of motion blocks in x and y direction handled by this cuda block */ \
    int nx = THREADSX_LOG2 - (xs_log2); \
    int ny = THREADSY_LOG2 - (ys_log2); \
    /** Motion block offset for this thread */ \
    int obx = ((bx) << nx) + (threadIdx.x >> (xs_log2)); \
    int oby = ((by) << ny) + (threadIdx.y >> (ys_log2)); \
    /** Block id base for this thread */ \
    int blockid = __mul24(oby, obmc.blocksx) + obx; \
    /** Offsets inside motion block */ \
    int ix = (threadIdx.x & ((1<<(xs_log2))-1)) << 1; \
    int iy = threadIdx.y & ((1<<(ys_log2))-1); \
    /** Offsets in image */ \
    int xofs = (obx<<xsep_log2) + ix + (xadd) + (1<<(xramp_log2-1)); \
    int yofs = (oby<<ysep_log2) + iy + (yadd) + (1<<(yramp_log2-1)); \
    /* Destination offset in memory */ \
    dest = OFFSET_U16(dest, __mul24(dstride, yofs) + (xofs<<1))

/* each thread calculates one pixel 
   Grid dimensions:
      x = div_roundup(max(xmid, xramp)*div_roundup(width, xsep), WIDTH_X) << 2
      y = div_roundup(max(ymid, yramp)*div_roundup(height, ysep), WIDTH_Y)
*/
// XXX process two pixels horizontally and write a 32 bit value
// XXX pad the blocks at the right and bottom so that this never runs out of the block array

// XXX handle the case in which xmid/ymid is 0 (complete overlap), or yramp/yramp is 0 (complete disjunct)
//     in this case, we need only 1 type of block instead of 4?

// XXX handle non power of two xsep/ysep

__global__ void motion_copy_2ref(
    u_int16_t* dest, int dstride, int dwidth, int dheight, int xB, int yB,
    // Component parameters
    int comp /* times 8 */, float sxscale /* exp2f(-x_shift) * 0.25f */ , float syscale /* exp2f(-y_shift) * 0.25f */,
    // Motion block parameters (depend on component too)
    int xramp_log2, int yramp_log2, int xsep_log2, int ysep_log2, int xmid_log2, int ymid_log2
)
{
    //__syncthreads(); 
    
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
    if(blockIdx.x < xB) 
    {
        if(blockIdx.y < yB) // 1: Inner (no blend)
        {
            STUFF(blockIdx.x, blockIdx.y, xmid_log2-1, ymid_log2, 0, 0);
            if(xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                int2 val;
                
                val = do_motion_block(blockid, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                
                /* Write value to memory */
                int16_2 sval;
                sval.x = val.x - 128;
                sval.y = val.y - 128;
                *((int16_2*)dest) = sval;
            }
        }
        else  // 3: Vertical (blend two blocks)
        {
            STUFF(blockIdx.x, blockIdx.y - yB, xmid_log2-1, yramp_log2, 0, 1<<ymid_log2);
            if(xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                int2 val;
                int yw2 = min(iy+1, (1<<yramp_log2));
                int yw1 = (1<<yramp_log2) - yw2;
                
                int2 valy1 = do_motion_block(blockid, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                int2 valy2 = do_motion_block(blockid + obmc.blocksx, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                
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
            STUFF(blockIdx.x - xB, blockIdx.y, xramp_log2-1, ymid_log2, 1<<xmid_log2, 0);
            if(xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                /* Sample the value */
                int2 val;
                int xw2 = min(ix+1, (1<<xramp_log2));
                int xw1 = (1<<xramp_log2) - xw2;
                
                int2 valx1 = do_motion_block(blockid, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                int2 valx2 = do_motion_block(blockid + 1, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                
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
            STUFF(blockIdx.x - xB, blockIdx.y - yB, xramp_log2-1, yramp_log2, 1<<xmid_log2, 1<<ymid_log2);
            if(xofs < dwidth && yofs < dheight) /* Clip to image */
            {
                int2 valy1,valy2;
                int xw2 = min(ix+1, (1<<xramp_log2));
                int xw1 = (1<<xramp_log2) - xw2;
                
                {
                    int2 valx1y1 = do_motion_block(blockid, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                    int2 valx2y1 = do_motion_block(blockid + 1, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                    valy1.x = __mul24(xw1, valx1y1.x) + __mul24(xw2, valx2y1.x);
                    valy1.y = __mul24(xw1, valx1y1.y) + __mul24(xw2, valx2y1.y);
                }
                {
                    int2 valx1y2 = do_motion_block(blockid + obmc.blocksx, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                    int2 valx2y2 = do_motion_block(blockid + obmc.blocksx + 1, xofs*PDIFF + POFS, yofs*PDIFF + POFS, comp, sxscale, syscale);
                    valy2.x = __mul24(xw1, valx1y2.x) + __mul24(xw2, valx2y2.x);
                    valy2.y = __mul24(xw1, valx1y2.y) + __mul24(xw2, valx2y2.y);
                }

                int2 val;
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

#undef motion_vectors
