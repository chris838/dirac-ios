#ifndef CUDAWL_FRAME
#define CUDAWL_FRAME

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_convert_u8_s16(uint8_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_u8_u8(uint8_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_u8_422_yuyv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_u8_422_uyvy(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_u8_444_ayuv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_yuyv_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_uyvy_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_convert_ayuv_u8_444 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_subtract_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_subtract_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_add_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);
extern void cuda_add_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
