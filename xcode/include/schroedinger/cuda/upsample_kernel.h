#define BLEFT 4 /* must be multiple of 4 */
#define BLEFT_U32 (BLEFT/4)
#define BRIGHT 4

// must be a multiple of 64
#define BS 256

// Assuming COLUMN_TILE_W and dataW are multiples
// of maximum coalescable read/write size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 64
#define COLUMN_TILE_W_SHIFT 6
#define COLUMN_TILE_H 24
#define COLUMN_THREAD_W (COLUMN_TILE_W/4)
#define COLUMN_THREAD_H 16

// Reading start offset for first block
#define COLUMN_FIRST_OFS (COLUMN_TILE_W * BLEFT)
#define COLUMN_LAST_OFS (COLUMN_TILE_W * (BLEFT + COLUMN_TILE_H))
#define COLUMN_OUT_OFS (COLUMN_TILE_W * (BLEFT + COLUMN_TILE_H + BRIGHT))
#define COLUMN_BLOCKSIZE (COLUMN_TILE_W * COLUMN_TILE_H)
#define COLUMN_ADVANCE (COLUMN_THREAD_H << COLUMN_TILE_W_SHIFT)
/// Shared mem size must be ROUNDUP4(BLEFT + width + BRIGHT) + width
/// stride must be a multiple of 4
__global__ void upsample_horizontal(uint8_t *output, int ostride, uint8_t *input, int istride, int width)
{
    extern __shared__ uint8_t data[];
    const int tid = threadIdx.x;
    
    // round up to multiple of four
    int out_ofs = BLEFT + width + BRIGHT;
    out_ofs = ROUNDUP4(out_ofs);

    // row offsets    
    input = OFFSET_U8(input, __mul24(blockIdx.x, istride));
    output = OFFSET_U8(output, __mul24(blockIdx.x, ostride));
    
    // read data from global, 4 bytes at a time
    int width_u32 = (width+3)>>2;
    int ofs;
    for(ofs=tid; ofs<width_u32; ofs+=BS)
        ((uint32_t*)data)[BLEFT_U32+ofs] = ((uint32_t*)input)[ofs];

    __syncthreads();

    // update left and right boundary conditions
    uint8_t left = data[BLEFT];
    uint8_t right = data[BLEFT+width-1];

    if((tid&3)==0 && (tid>>2) < BLEFT)
        data[(tid>>2)] = left;
    if((tid&3)==1 && (tid>>2) < BRIGHT)
        data[BLEFT+width+(tid>>2)] = right;
  
    __syncthreads();
    
    // do the convolution
    // how to distribute this over threads as to that we do not get bank conflicts
    // there are 16 banks, so a logical distribution would be
    // how to convert thread id into offset? some bit reordering of upper 6 bits
    //int tid_u8 = ((tid&3)<<4)|((tid&63)>>2)|(tid&~63);
    int tid_u8 = ((tid&15)<<2)|((tid&63)>>4)|(tid&~63);
    for(ofs=tid_u8; ofs<width; ofs+=BS)
    {
        int val = 16;
        val += __mul24(-1, data[BLEFT+ofs-3]);
        val += __mul24(3, data[BLEFT+ofs-2]);
        val += __mul24(-7, data[BLEFT+ofs-1]);
        val += __mul24(21, data[BLEFT+ofs]);
        val += __mul24(21, data[BLEFT+ofs+1]);
        val += __mul24(-7, data[BLEFT+ofs+2]);
        val += __mul24(3, data[BLEFT+ofs+3]);
        val += __mul24(-1, data[BLEFT+ofs+4]);
        
        data[out_ofs + ofs] = val >> 5;
    }
    __syncthreads();
    
    // now write back, interleave data[BLEFT + ofs] and data[out_ofs + ofs] in ABABABABAB order
    // write in units of 4 bytes, of course
    // again, assume stride is a multiple of 4
    width_u32 = ((width*2)+3)>>2;
    
    for(ofs=tid; ofs<width_u32; ofs+=BS)
    {
        // two-way bank conflict unavoidable?
        // maybe not if you would write 8 bytes per thread,
        // and thus read 2x4 from shared. This would assume stride is a multiply of 8, which is
        // a fair requirement: in practice, it is a multiple of 64
        ((uint32_t*)output)[ofs] =
            (data[BLEFT + (ofs<<1) + 0]<<0)|
            (data[BLEFT + (ofs<<1) + 1]<<16)|
            (data[out_ofs + (ofs<<1) + 0]<<8)|
            (data[out_ofs + (ofs<<1) + 1]<<24);
    }
}
// 3, -11, 25, -56, 167, 167, -56, 25, -11, 3
__device__ void convolute(int offset)
{
    extern __shared__ uint8_t data[];
    data[COLUMN_OUT_OFS+offset] = 
        (16+
        __mul24(-1, data[COLUMN_FIRST_OFS+offset-3*COLUMN_TILE_W])+
        __mul24(3, data[COLUMN_FIRST_OFS+offset-2*COLUMN_TILE_W])+
        __mul24(-7, data[COLUMN_FIRST_OFS+offset-1*COLUMN_TILE_W])+
        __mul24(21, data[COLUMN_FIRST_OFS+offset-0*COLUMN_TILE_W])+
        __mul24(21, data[COLUMN_FIRST_OFS+offset+1*COLUMN_TILE_W])+
        __mul24(-7, data[COLUMN_FIRST_OFS+offset+2*COLUMN_TILE_W])+
        __mul24(3, data[COLUMN_FIRST_OFS+offset+3*COLUMN_TILE_W])+
        __mul24(-1, data[COLUMN_FIRST_OFS+offset+4*COLUMN_TILE_W]))>>5;
}

/// Shared mem size must be COLUMN_TILE_W*(BLEFT+COLUMN_TILE_H+BRIGHT) + COLUMN_TILE_W*COLUMN_TILE_H
__global__ void upsample_vertical(uint8_t *output, int ostride, uint8_t *input, int istride, int width, int height)
{
    extern __shared__ uint8_t data[];

    /// starting y of block
    int gyofs_out = __mul24(blockIdx.y, COLUMN_TILE_H);
    int gyofs_in = max(gyofs_out - BLEFT, 0);
    /// starting x of block
    int gxofs = blockIdx.x << COLUMN_TILE_W_SHIFT;
    /// starting offset in global memory (input)
    input = OFFSET_U8(input, __mul24(gyofs_in + threadIdx.y, istride) + gxofs + (threadIdx.x<<2));
    /// starting offset in global memory (output)
    output = OFFSET_U8(output, __mul24(gyofs_out + threadIdx.y, ostride) + gxofs + (threadIdx.x<<2));
    /// number of lines to read
    int lines_r = min(height - gyofs_in, BLEFT + COLUMN_TILE_H + BRIGHT);
    /// number of lines to write
    int lines_w = min(height - gyofs_out, COLUMN_TILE_H);
    /// number of columns to process, in units of four
    int clampwidth = (min(width - gxofs, COLUMN_TILE_W)+3)>>2;
    /// shared memory element for me
    int selement = (threadIdx.x<<2) + (threadIdx.y << COLUMN_TILE_W_SHIFT);

    /// Read from global, fill shared mem
    int tofs;
    if(blockIdx.y==0)
        tofs = COLUMN_FIRST_OFS;
    else
        tofs = 0;
    int send = tofs + (lines_r << COLUMN_TILE_W_SHIFT);
    
    if(threadIdx.x < clampwidth)
    {
        int my_stride = __mul24(COLUMN_THREAD_H, istride);

        int gofs = 0;
        int sofs = tofs + selement;
        for(; sofs<send; sofs += COLUMN_ADVANCE, gofs += my_stride)
            *((uint32_t*)&data[sofs]) = *((uint32_t*)&input[gofs]);
    }

    __syncthreads();
    /// boundary conditions
    if(blockIdx.y == 0)
    {
        /// Fill up top BLEFT rows with first row
        if(selement < (BLEFT << COLUMN_TILE_W_SHIFT))
            *((uint32_t*)&data[selement]) = *((uint32_t*)&data[COLUMN_FIRST_OFS + (threadIdx.x<<2)]);
    }
    if(blockIdx.y == (gridDim.y-1))
    {
        /// Fill BRIGHT rows
        if((selement+send) < COLUMN_OUT_OFS)
            *((uint32_t*)&data[selement+send]) = *((uint32_t*)&data[send - COLUMN_TILE_W + (threadIdx.x<<2)]);
    }

    __syncthreads();

    /// Process
    for(int sofs=selement; sofs<COLUMN_BLOCKSIZE; sofs+=COLUMN_ADVANCE)
    {
        /// Each thread must process four 8-bit elements
        convolute(sofs);
        convolute(sofs+1);
        convolute(sofs+2);
        convolute(sofs+3);
    }

    __syncthreads();
    
    /// Write to output
    if(threadIdx.x < clampwidth)
    {
        int sofs = COLUMN_OUT_OFS;
        int send = COLUMN_OUT_OFS + (lines_w << COLUMN_TILE_W_SHIFT);
        int my_stride = __mul24(COLUMN_THREAD_H, ostride);

        int gofs = 0;       
        sofs += selement;
        for(; sofs<send; sofs += COLUMN_ADVANCE, gofs += my_stride)
            *((uint32_t*)&output[gofs]) = *((uint32_t*)&data[sofs]);
    }
}
