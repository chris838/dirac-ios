
#ifndef __SCHRO_OPENGL_WAVELET_H__
#define __SCHRO_OPENGL_WAVELET_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

void schro_opengl_wavelet_transform (SchroFrameData *frame_data, int filter);
void schro_opengl_wavelet_vertical_deinterleave (SchroFrameData *frame_data);
void schro_opengl_wavelet_inverse_transform (SchroFrameData *frame_data,
    int filter);

SCHRO_END_DECLS

#endif

