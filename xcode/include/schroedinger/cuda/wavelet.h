#include "wavelets.h"
#define OFFSET(ptr,offset) ((int16_t *)(((uint8_t *)(ptr)) + (offset)))
void
schro_split_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 7, 4 };
  static const int16_t stage2_offset_shift[] = { 16, 5 };

  /// Boundary conditions
  /// Left 1
  hi[-1] = hi[0];
  /// Right 2
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  oil_mas4_add_s16 (lo, lo, hi-1, stage1_weights, stage1_offset_shift, n);

  /// Boundary conditions
  /// Left 2
  lo[-1] = lo[0];
  lo[-2] = lo[0];
  /// Right 1
  lo[n] = lo[n-1];

  oil_mas4_add_s16 (hi, hi, lo - 2, stage2_weights, stage2_offset_shift, n);
}

void schro_iwt_13_5 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  int i;
  int16_t one = 1;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 8;i++){
    int i1 = i-4;
    int i2 = i-6;
#ifdef HORIZONTAL
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      oil_lshift_s16(ROW(i), ROW(i), &one, width);
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_135 (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
#endif
#ifdef VERTICAL
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 7, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { -8, -9, 1, 0 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-4) {
        static const int16_t stage1_weights[] = { 0, 1, -9, -8 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-4), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-2) {
        static const int16_t stage1_weights[] = { 1, -17 };
        oil_mas2_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-2), ROW(i1),
            stage1_weights, stage1_offset_shift, width);
      } else {
        static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-2), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 16, 5 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { 17, -1 };
        oil_mas2_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2+1), ROW(i2+3),
            stage2_weights, stage2_offset_shift, width);
      } else if (i2 == 2) {
        static const int16_t stage2_weights[] = { 8, 9, -1, 0 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-1), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      } else if (i2 == height-2) {
        static const int16_t stage2_weights[] = { 0, -1, 9, 8 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-5), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      } else {
        static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-3), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      }
    }
#endif
#undef ROW
  }
}



void 
schro_split_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 0, 1 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);
}

void schro_iwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 0, 1 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };
  int i;
  int16_t one = 1;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 2;i++){
#ifdef HORIZONTAL
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      oil_lshift_s16(ROW(i), ROW(i), &one, width);
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_53 (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
#endif
#ifdef VERTICAL
    if ((i&1) == 0 && i >= 2) {
      int16_t *d;
      if (i<height) {
        d = OFFSET(data,i*stride);
      } else {
        d = OFFSET(data,(height-2)*stride);
      }
      oil_mas2_across_add_s16 (
          OFFSET(data, (i-1)*stride),
          OFFSET(data, (i-1)*stride),
          OFFSET(data, (i-2)*stride),
          d,
          stage1_weights, stage1_offset_shift, width);

      if (i-3>=0) {
        d = OFFSET(data, (i-3)*stride);
      } else {
        d = OFFSET(data, 1*stride);
      }
      oil_mas2_across_add_s16 (
          OFFSET(data, (i-2)*stride),
          OFFSET(data, (i-2)*stride),
          d,
          OFFSET(data, (i-1)*stride),
          stage2_weights, stage2_offset_shift, width);
    }
#endif
  }
#undef ROW
}

void schro_iwt_5_3_h (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;
  int16_t one = 1;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height+2;i++){
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      
      oil_lshift_s16(ROW(i), ROW(i), &one, width);
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_53 (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
    
  }
#undef ROW
}


void
schro_split_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 7, 4 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };

  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);
}

void schro_iwt_desl_9_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;
  int16_t one = 1;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))

  for(i=0;i<height + 6;i++){
    int i1 = i-4;
    int i2 = i-6;
#ifdef HORIZONTAL
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      oil_lshift_s16(ROW(i), ROW(i), &one, width);
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_desl93 (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
#endif
#ifdef VERTICAL
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 7, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { -8, -9, 1, 0 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, i1*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-4) {
        static const int16_t stage1_weights[] = { 0, 1, -9, -8 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, (i1-4)*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-2) {
        static const int16_t stage1_weights[] = { 1, -17 };
        oil_mas2_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, (i1-2)*stride), OFFSET(data, i1*stride),
            stage1_weights, stage1_offset_shift, width);
      } else {
        static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data,(i1+1)*stride),
            OFFSET(data,(i1-2)*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 2, 2 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { 1, 1 };
        oil_mas2_across_add_s16 (
            OFFSET(data,i2*stride), OFFSET(data, i2*stride),
            OFFSET(data, (i2+1)*stride), OFFSET(data, (i2+1)*stride),
            stage2_weights, stage2_offset_shift, width);
      } else {
        static const int16_t stage2_weights[] = { 1, 1 };
        oil_mas2_across_add_s16 (
            OFFSET(data,i2*stride), OFFSET(data, i2*stride),
            OFFSET(data, (i2-1)*stride), OFFSET(data, (i2+1)*stride),
            stage2_weights, stage2_offset_shift, width);
      }
    }
#endif
  }
}


void
schro_split_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -6497, -6497 };
  static const int16_t stage2_weights[] = { -217, -217 };
  static const int16_t stage3_weights[] = { 3616, 3616 };
  static const int16_t stage4_weights[] = { 1817, 1817 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };

  hi[-1] = hi[0]; //?
  hi[n] = hi[n-1]; 

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1]; // ?

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage12_offset_shift, n);

  hi[-1] = hi[0]; // ?
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage3_weights, stage34_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1]; // ?

  oil_mas2_add_s16 (hi, hi, lo - 1, stage4_weights, stage34_offset_shift, n);
}

void schro_iwt_daub_9_7 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  static const int16_t stage1_weights[] = { -6497, -6497 };
  static const int16_t stage2_weights[] = { -217, -217 };
  static const int16_t stage3_weights[] = { 3616, 3616 };
  static const int16_t stage4_weights[] = { 1817, 1817 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };
  int i;
  int16_t one = 1;
  int i1;
  int i2;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 4;i++){
    i1 = i - 2;
    i2 = i - 4;
#ifdef HORIZONTAL
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      oil_lshift_s16(ROW(i), ROW(i), &one, width);
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_daub97 (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
#endif
#ifdef VERTICAL
    if ((i1&1) == 0 && i1 >=0 && i1 < height) {
      int16_t *d;
      if (i1+2<height) {
        d = ROW(i1+2);
      } else {
        d = ROW(height-2);
      }
      oil_mas2_across_add_s16 (ROW(i1+1), ROW(i1+1), ROW(i1), d,
          stage1_weights, stage12_offset_shift, width);

      if (i1-1>=0) {
        d = ROW(i1-1);
      } else {
        d = ROW(1);
      }
      oil_mas2_across_add_s16 (ROW(i1), ROW(i1), d, ROW(i1+1),
          stage2_weights, stage12_offset_shift, width);
    }
    if ((i2&1) == 0 && i2 >=0 && i2 < height) {
      int16_t *d;
      if (i2+2<height) {
        d = ROW(i2+2);
      } else {
        d = ROW(height-2);
      }
      oil_mas2_across_add_s16 (ROW(i2+1), ROW(i2+1), ROW(i2), d,
          stage3_weights, stage34_offset_shift, width);

      if (i2-1>=0) {
        d = ROW(i2-1);
      } else {
        d = ROW(1);
      }
      oil_mas2_across_add_s16 (ROW(i2), ROW(i2), d, ROW(i2+1),
          stage4_weights, stage34_offset_shift, width);
    }
#endif
  }
#undef ROW
}


void
schro_split_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

  for(i=0;i<n;i++) {
    lo[i] -= hi[i];
  }
  for(i=0;i<n;i++) {
    hi[i] += ((lo[i] + 1)>>1);
  }
}

static void
schro_iwt_haar (int16_t *data, int stride, int width, int height,
    int16_t *tmp, int16_t shift)
{
  int16_t *data1;
  int16_t *data2;
  int i;
  int j;

  for(i=0;i<height;i+=2){
    data1 = OFFSET(data,i*stride);
    data2 = OFFSET(data,(i+1)*stride);
#ifdef HORIZONTAL
    if (shift) {
      oil_lshift_s16(tmp, data1, &shift, width);
    } else {
      oil_memcpy (tmp, data1, width*sizeof(int16_t));
    }
    oil_deinterleave2_s16 (data1, data1 + width/2, tmp, width/2);
    schro_split_ext_haar (data1, data1 + width/2, width/2);
    
    if (shift) {
      oil_lshift_s16(tmp, data2, &shift, width);
    } else {
      oil_memcpy (tmp, data2, width*sizeof(int16_t));
    }
    oil_deinterleave2_s16 (data2, data2 + width/2, tmp, width/2);
    schro_split_ext_haar (data2, data2 + width/2, width/2);
#endif
#ifdef VERTICAL
    for(j=0;j<width;j++){
      data2[j] -= data1[j];
      data1[j] += (data2[j] + 1)>>1;
    }
#endif
  }
}


void
schro_split_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static const int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
  static const int16_t stage1_offset_shift[] = { 128, 8 };
  static const int16_t stage2_offset_shift[] = { 127, 8 };

  lo[-4] = lo[0];
  lo[-3] = lo[0];
  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];
  lo[n+2] = lo[n-1];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage1_weights, stage1_offset_shift, n);

  hi[-3] = hi[0];
  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  hi[n+2] = hi[n-1];
  hi[n+3] = hi[n-1];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage2_weights, stage2_offset_shift, n);
}

void schro_iwt_fidelity (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 16;i++){
    int i1 = i-8;
    int i2 = i-16;
#ifdef HORIZONTAL
    if (i < height) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + 12 + width/2;
      oil_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_fidelity (hi, lo, width/2);
      oil_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      oil_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
#endif
#ifdef VERTICAL
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 128, 8 };
      static const int16_t stage1_weights[][8] = {
        { 161 + 161 - 46 + 21 - 8, -46, 21, -8, 0, 0, 0, 0 },
        { 161 - 46 + 21 - 8, 161, -46, 21, -8, 0, 0, 0 },
        { -46 + 21 - 8, 161, 161, -46, 21, -8, 0, 0 },
        { 21 - 8, -46, 161, 161, -46, 21, -8, 0 },
        { -8, 21, -46, 161, 161, -46, 21, -8 },
        { 0, -8, 21, -46, 161, 161, -46, 21 -8 },
        { 0, 0, -8, 21, -46, 161, 161, -46 + 21 - 8 },
        { 0, 0, 0, -8, 21, -46, 161, 161 - 46 + 21 - 8 },
      };
      const int16_t *weights;
      int offset;
      if (i1 < 8) {
        weights = stage1_weights[i1/2];
        offset = 1;
      } else if (i1 >= height - 6) {
        weights = stage1_weights[8 - (height - i1)/2];
        offset = height + 1 - 16;
      } else {
        weights = stage1_weights[4];
        offset = i1 - 7;
      }
      oil_mas8_across_add_s16 (
          ROW(i1), ROW(i1), ROW(offset), stride * 2,
          weights, stage1_offset_shift, width);
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 127, 8 };
      static const int16_t stage2_weights[][8] = {
        { 2 - 10 + 25 - 81, -81, 25, -10, 2, 0, 0, 0 },
        { 2 - 10 + 25, -81, -81, 25, -10, 2, 0, 0 },
        { 2 -10, 25, -81, -81, 25, -10, 2, 0 },
        { 2, -10, 25, -81, -81, 25, -10, 2 },
        { 0, 2, -10, 25, -81, -81, 25, -10 + 2 },
        { 0, 0, 2, -10, 25, -81, -81, 25 - 10 + 2 },
        { 0, 0, 0, 2, -10, 25, -81, -81 + 25 - 10 + 2 },
        { 0, 0, 0, 0, 2, -10, 25, -81 - 81 + 25 - 10 + 2 }
      };
      const int16_t *weights;
      int offset;
      if (i2 < 6) {
        weights = stage2_weights[i2/2];
        offset = 0;
      } else if (i2 >= height - 8) {
        weights = stage2_weights[8 - (height - i2)/2];
        offset = height - 16;
      } else {
        weights = stage2_weights[3];
        offset = i2 - 6;
      }
      oil_mas8_across_add_s16 (
          ROW(i2+1), ROW(i2+1), ROW(offset), stride * 2,
          weights, stage2_offset_shift, width);
    }
#endif
  }
#undef ROW
}

void
schro_synth_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 1, 2 };
  static const int16_t stage2_offset_shift[] = { 8, 4 };

  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 1, 2 };
  static const int16_t stage2_offset_shift[] = { 1, 1 };

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 15, 5 };
  static const int16_t stage2_offset_shift[] = { 8, 4 };

  lo[-1] = lo[0];
  lo[-2] = lo[0];
  lo[n] = lo[n-1];
  oil_mas4_add_s16 (hi, hi, lo - 2, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  oil_mas4_add_s16 (lo, lo, hi-1, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

  for(i=0;i<n;i++) {
    hi[i] -= ((lo[i] + 1)>>1);
  }
  for(i=0;i<n;i++) {
    lo[i] += hi[i];
  }
}

void
schro_synth_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static const int16_t stage2_weights[] = { 8, -21, 46, -161, -161, 46, -21, 8 };
  static const int16_t stage1_offset_shift[] = { 128, 8 };
  static const int16_t stage2_offset_shift[] = { 127, 8 };

  hi[-3] = hi[0];
  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  hi[n+2] = hi[n-1];
  hi[n+3] = hi[n-1];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage1_weights, stage1_offset_shift, n);

  lo[-4] = lo[0];
  lo[-3] = lo[0];
  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];
  lo[n+2] = lo[n-1];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1817, -1817 };
  static const int16_t stage2_weights[] = { -3616, -3616 };
  static const int16_t stage3_weights[] = { 217, 217 };
  static const int16_t stage4_weights[] = { 6497, 6497 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage12_offset_shift, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage3_weights, stage34_offset_shift, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage4_weights, stage34_offset_shift, n);
}



void schro_iiwt_desl_9_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))

  for(i=-6;i<height;i++){
    int i1 = i+2;
    int i2 = i+6;
#ifdef VERTICAL
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 1, 2 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { -1, -1 };
        oil_mas2_across_add_s16 (
            OFFSET(data,i2*stride), OFFSET(data, i2*stride),
            OFFSET(data, (i2+1)*stride), OFFSET(data, (i2+1)*stride),
            stage2_weights, stage2_offset_shift, width);
      } else {
        static const int16_t stage2_weights[] = { -1, -1 };
        oil_mas2_across_add_s16 (
            OFFSET(data,i2*stride), OFFSET(data, i2*stride),
            OFFSET(data, (i2-1)*stride), OFFSET(data, (i2+1)*stride),
            stage2_weights, stage2_offset_shift, width);
      }
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 8, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { 8, 9, -1, 0 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, i1*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-4) {
        static const int16_t stage1_weights[] = { 0, -1, 9, 8 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, (i1-4)*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-2) {
        static const int16_t stage1_weights[] = { -1, 17 };
        oil_mas2_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data, (i1+1)*stride),
            OFFSET(data, (i1-2)*stride), OFFSET(data, i1*stride),
            stage1_weights, stage1_offset_shift, width);
      } else {
        static const int16_t stage1_weights[] = { -1, 9, 9, -1 };
        oil_mas4_across_add_s16 (
            OFFSET(data,(i1+1)*stride), OFFSET(data,(i1+1)*stride),
            OFFSET(data,(i1-2)*stride), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      }
    }
#endif
#ifdef HORIZONTAL
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      static const int16_t as[2] = { 1, 1 };
      oil_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      oil_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_desl93 (hi, lo, width/2);
      oil_interleave2_s16 (ROW(i), hi, lo, width/2);
      oil_add_const_rshift_s16(ROW(i), ROW(i), as, width);
    }
#endif
  }
}

void schro_iiwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  static const int16_t stage1_weights[] = { 1, 1 };
  static const int16_t stage2_weights[] = { -1, -1 };
  static const int16_t stage1_offset_shift[] = { 1, 1 };
  static const int16_t stage2_offset_shift[] = { 1, 2 };
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-4;i<height + 2;i++){
    int i1 = i + 2;
    int i2 = i + 4;
#ifdef VERTICAL
    if ((i2&1) == 0 && i2 >= 0 && i2 < height) {
      int16_t *d;
      if (i2-1>=0) {
        d = OFFSET(data, (i2-1)*stride);
      } else {
        d = OFFSET(data, 1*stride);
      }
      oil_mas2_across_add_s16 (
          OFFSET(data, i2*stride),
          OFFSET(data, i2*stride),
          d,
          OFFSET(data, (i2+1)*stride),
          stage2_weights, stage2_offset_shift, width);
    }
    if ((i1&1) == 0 && i1 >= 0 && i1 < height) {
      int16_t *d;
      if (i1+2<height) {
        d = OFFSET(data,(i1+2)*stride);
      } else {
        d = OFFSET(data,(height-2)*stride);
      }
      oil_mas2_across_add_s16 (
          OFFSET(data, (i1+1)*stride),
          OFFSET(data, (i1+1)*stride),
          OFFSET(data, i1*stride),
          d,
          stage1_weights, stage1_offset_shift, width);
    } 
#endif
#ifdef HORIZONTAL
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      static const int16_t as[2] = { 1, 1 };
      oil_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      oil_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_53 (hi, lo, width/2);
      oil_interleave2_s16 (ROW(i), hi, lo, width/2);
      oil_add_const_rshift_s16(ROW(i), ROW(i), as, width);
    }
#endif
  }
#undef ROW
}

void schro_iiwt_13_5 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-8;i<height;i++){
    int i1 = i+4;
    int i2 = i+8;
#ifdef VERTICAL
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 15, 5 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { -17, 1 };
        oil_mas2_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2+1), ROW(i2+3),
            stage2_weights, stage2_offset_shift, width);
      } else if (i2 == 2) {
        static const int16_t stage2_weights[] = { -8, -9, 1, 0 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-1), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      } else if (i2 == height-2) {
        static const int16_t stage2_weights[] = { 0, 1, -9, -8 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-5), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      } else {
        static const int16_t stage2_weights[] = { 1, -9, -9, 1 };
        oil_mas4_across_add_s16 (
            ROW(i2), ROW(i2), ROW(i2-3), stride * 2,
            stage2_weights, stage2_offset_shift, width);
      }
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 8, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { 8, 9, -1, 0 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-4) {
        static const int16_t stage1_weights[] = { 0, -1, 9, 8 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-4), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      } else if (i1 == height-2) {
        static const int16_t stage1_weights[] = { -1, 17 };
        oil_mas2_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-2), ROW(i1),
            stage1_weights, stage1_offset_shift, width);
      } else {
        static const int16_t stage1_weights[] = { -1, 9, 9, -1 };
        oil_mas4_across_add_s16 (
            ROW(i1+1), ROW(i1+1), ROW(i1-2), stride * 2,
            stage1_weights, stage1_offset_shift, width);
      }
    }
#endif
#ifdef HORIZONTAL
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      static const int16_t as[2] = { 1, 1 };
      oil_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      oil_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_135 (hi, lo, width/2);
      oil_interleave2_s16 (ROW(i), hi, lo, width/2);
      oil_add_const_rshift_s16(ROW(i), ROW(i), as, width);
    }
#endif
#undef ROW
  }
}

static void
schro_iiwt_haar (int16_t *data, int stride, int width, int height,
    int16_t *tmp, int16_t shift)
{
  int16_t *data1;
  int16_t *data2;
  int i;
  int j;
  int16_t as[2];
  
  as[0] = (1<<shift)>>1;
  as[1] = shift;

  for(i=0;i<height;i+=2){
    data1 = OFFSET(data,i*stride);
    data2 = OFFSET(data,(i+1)*stride);
#ifdef VERTICAL
    for(j=0;j<width;j++){
      data1[j] -= (data2[j] + 1)>>1;
      data2[j] += data1[j];
    }
#endif
#ifdef HORIZONTAL
    schro_synth_ext_haar (data1, data1 + width/2, width/2);
    if (shift) {
      oil_add_const_rshift_s16(tmp, data1, as, width);
    } else {
      oil_memcpy (tmp, data1, width*sizeof(int16_t));
    }
    oil_interleave2_s16 (data1, tmp, tmp + width/2, width/2);

    schro_synth_ext_haar (data2, data2 + width/2, width/2);
    if (shift) {
      oil_add_const_rshift_s16(tmp, data2, as, width);
    } else {
      oil_memcpy (tmp, data2, width*sizeof(int16_t));
    }
    oil_interleave2_s16 (data2, tmp, tmp + width/2, width/2);
#endif
  }
}

void schro_iiwt_haar0 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iiwt_haar (data, stride, width, height, tmp, 0);
}

void schro_iiwt_haar1 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iiwt_haar (data, stride, width, height, tmp, 1);
}

void schro_iiwt_haar2 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iiwt_haar (data, stride, width, height, tmp, 2);
}

void schro_iiwt_fidelity (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

  /* FIXME */
//  SCHRO_ASSERT(height>=16);

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-16;i<height;i++){
    int i1 = i+8;
    int i2 = i+16;
#ifdef VERTICAL
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_offset_shift[] = { 128, 8 };
      static const int16_t stage2_weights[][8] = {
        { -2 + 10 - 25 + 81, 81, -25, 10, -2, 0, 0, 0 },
        { -2 + 10 - 25, 81, 81, -25, 10, -2, 0, 0 },
        { -2 + 10, -25, 81, 81, -25, 10, -2, 0 },
        { -2, 10, -25, 81, 81, -25, 10, -2 },
        { 0, -2, 10, -25, 81, 81, -25, 10 - 2 },
        { 0, 0, -2, 10, -25, 81, 81, -25 + 10 - 2 },
        { 0, 0, 0, -2, 10, -25, 81, 81 - 25 + 10 - 2 },
        { 0, 0, 0, 0, -2, 10, -25, 81 + 81 - 25 + 10 - 2 }
      };
      const int16_t *weights;
      int offset;
      if (i2 < 6) {
        weights = stage2_weights[i2/2];
        offset = 0;
      } else if (i2 >= height - 8) {
        weights = stage2_weights[8 - (height - i2)/2];
        offset = height - 16;
      } else {
        weights = stage2_weights[3];
        offset = i2 - 6;
      }
      oil_mas8_across_add_s16 (
          ROW(i2+1), ROW(i2+1), ROW(offset), stride * 2,
          weights, stage2_offset_shift, width);
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_offset_shift[] = { 127, 8 };
      static const int16_t stage1_weights[][8] = {
        { 8 - 21 + 46 - 161 - 161, 46, -21, 8, 0, 0, 0, 0 },
        { 8 - 21 + 46 - 161, -161, 46, -21, 8, 0, 0, 0 },
        { 8 - 21 + 46, -161, -161, 46, -21, 8, 0, 0 },
        { 8 -21, 46, -161, -161, 46, -21, 8, 0 },
        { 8, -21, 46, -161, -161, 46, -21, 8 },
        { 0, 8, -21, 46, -161, -161, 46, -21 + 8 },
        { 0, 0, 8, -21, 46, -161, -161, 46 - 21 + 8 },
        { 0, 0, 0, 8, -21, 46, -161, -161 + 46 - 21 + 8 },
      };
      const int16_t *weights;
      int offset;
      if (i1 < 8) {
        weights = stage1_weights[i1/2];
        offset = 1;
      } else if (i1 >= height - 6) {
        weights = stage1_weights[8 - (height - i1)/2];
        offset = height + 1 - 16;
      } else {
        weights = stage1_weights[4];
        offset = i1 - 7;
      }
      oil_mas8_across_add_s16 (
          ROW(i1), ROW(i1), ROW(offset), stride * 2,
          weights, stage1_offset_shift, width);
    }
#endif
#ifdef HORIZONTAL
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + 12 + width/2;
      oil_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      oil_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_fidelity (hi, lo, width/2);
      oil_interleave2_s16 (ROW(i), hi, lo, width/2);
    }
#endif
  }
#undef ROW
}

void schro_iiwt_daub_9_7 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  static const int16_t stage1_weights[] = { 6497, 6497 };
  static const int16_t stage2_weights[] = { 217, 217 };
  static const int16_t stage3_weights[] = { -3616, -3616 };
  static const int16_t stage4_weights[] = { -1817, -1817 };
  static const int16_t stage12_offset_shift[] = { 2048, 12 };
  static const int16_t stage34_offset_shift[] = { 2047, 12 };
  int i;
  int i1;
  int i2;
  int i3;
  int i4;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-6;i<height;i++){
    i1 = i + 0;
    i2 = i + 2;
    i3 = i + 2;
    i4 = i + 6;
#ifdef VERTICAL
    if ((i4&1) == 0 && i4 >=0 && i4 < height) {
      int16_t *d;
      if (i4-1>=0) {
        d = ROW(i4-1);
      } else {
        d = ROW(1);
      }
      oil_mas2_across_add_s16 (ROW(i4), ROW(i4), d, ROW(i4+1),
          stage4_weights, stage34_offset_shift, width);
    }

    if ((i3&1) == 0 && i3 >=0 && i3 < height) {
      int16_t *d;
      if (i3+2<height) {
        d = ROW(i3+2);
      } else {
        d = ROW(height-2);
      }
      oil_mas2_across_add_s16 (ROW(i3+1), ROW(i3+1), ROW(i3), d,
          stage3_weights, stage34_offset_shift, width);
    }

    if ((i2&1) == 0 && i2 >=0 && i2 < height) {
      int16_t *d;

      if (i2-1>=0) {
        d = ROW(i2-1);
      } else {
        d = ROW(1);
      }
      oil_mas2_across_add_s16 (ROW(i2), ROW(i2), d, ROW(i2+1),
          stage2_weights, stage12_offset_shift, width);
    }

    if ((i1&1) == 0 && i1 >=0 && i1 < height) {
      int16_t *d;
      if (i1+2<height) {
        d = ROW(i1+2);
      } else {
        d = ROW(height-2);
      }
      oil_mas2_across_add_s16 (ROW(i1+1), ROW(i1+1), ROW(i1), d,
          stage1_weights, stage12_offset_shift, width);
    }
#endif
#ifdef HORIZONTAL
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      static const int16_t as[2] = { 1, 1 };
      oil_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      oil_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_daub97 (hi, lo, width/2);
      oil_interleave2_s16 (ROW(i), hi, lo, width/2);
      oil_add_const_rshift_s16(ROW(i), ROW(i), as, width);
    }
#endif
  }
#undef ROW
}

void schro_iwt_haar0 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 0);
}

void schro_iwt_haar1 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 1);
}

void schro_iwt_haar2 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 2);
}


void
schro_wavelet_inverse_transform_2d (int filter, int16_t *data, int stride,
    int width, int height, int16_t *tmp)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_iiwt_desl_9_3 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_5_3:
      schro_iiwt_5_3 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_13_5:
      schro_iiwt_13_5 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iiwt_haar0 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iiwt_haar1 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_2:
      schro_iiwt_haar2 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iiwt_fidelity (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_iiwt_daub_9_7(data, stride, width, height, tmp);
      break;
  }
}

void
schro_wavelet_transform_2d (int filter, int16_t *data, int stride, int width,
    int height, int16_t *tmp)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_iwt_desl_9_3 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_5_3:
      schro_iwt_5_3 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_13_5:
      schro_iwt_13_5 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iwt_haar0 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iwt_haar1 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_2:
      schro_iwt_haar2 (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iwt_fidelity (data, stride, width, height, tmp);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_iwt_daub_9_7(data, stride, width, height, tmp);
      break;
  }
}
