#ifndef H_CUDA_MOTION
#define H_CUDA_MOTION

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _CudaMotion CudaMotion;

CudaMotion *cuda_motion_init(cudaStream_t stream);
void cuda_motion_free(CudaMotion *rv);

/** OBMC parameters */
struct _Obmc {
    int blocksx;
    int blocksy;

    //int shift;  // log2(x_ramp) + log2(y_ramp)
    
    // For luma
    int x_ramp; // (x_len-x_sep)
    int y_ramp; // (y_len-y_sep)
    int x_len;
    int y_len;
    int x_sep;
    int y_sep;
    int x_mid; // (x_sep-x_ramp)
    int y_mid; // (y_sep-y_ramp)

    // Values converted to powers of two for efficient arithmetic
    // If the value is 0 or another non-power of two, these should be -1
    int x_ramp_log2;
    int y_ramp_log2;
    int x_sep_log2;
    int y_sep_log2;
    int x_mid_log2;
    int y_mid_log2;
    
    int weight1;
    int weight2;
    int weight_shift;
    
    /* number of precision bits */
    int mv_precision;
};


/** Global motion vector */
struct _GlobalMotion {
    int shift;
    int b0;
    int b1;
    int a_exp;
    int a00;
    int a01;
    int a10;
    int a11;
    int c_exp;
    int c0;
    int c1;
};

/** Local motion vectors */

/* Specify that this vector is unused */
#define MOTION_NONE 0x7FFE   
/* Specify global motion for this block */
#define MOTION_GLOBAL 0x7FFD 
struct __align__((8)) _MotionVector {
    int16_t x1;
    int16_t x2;
    int16_t y1;
    int16_t y2;
};

struct _MotionData
{
    /// Parameters
    struct _Obmc obmc;
};

typedef struct _MotionData CudaMotionData;

/// Reserve space for N vectors
struct _MotionVector *cuda_motion_reserve(CudaMotion *self, int width, int height);

void cuda_motion_begin(CudaMotion *self, CudaMotionData *d);
void cuda_motion_copy(CudaMotion *self, CudaMotionData *d, int16_t *output, int ostride, int width, int height, int component, int xshift, int yshift, struct cudaArray *aref1, struct cudaArray *aref2);

#ifdef __cplusplus
}
#endif


#endif
