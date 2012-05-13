
#ifndef __SCHRO_OPENGL_CANVAS_H__
#define __SCHRO_OPENGL_CANVAS_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/opengl/schroopengltypes.h>
#include <GL/glew.h>

SCHRO_BEGIN_DECLS

#define SCHRO_OPENGL_TRANSFER_PIXELBUFFERS 4

/*typedef enum _SchroOpenGLTextureFormat {
  SCHRO_OPENGL_TEXTURE_FORMAT_FIXED_POINT = 0,
  SCHRO_OPENGL_TEXTURE_FORMAT_FLOAT,
  SCHRO_OPENGL_TEXTURE_FORMAT_INTEGER
} SchroOpenGLTextureFormat;*/

struct _SchroOpenGLTexture {
  GLuint handles[2];
  //SchroOpenGLTextureFormat format;
  //unsigned int flags;
  GLenum internal_format;
  GLenum pixel_format;
  GLenum type;
  int channels;
};

struct _SchroOpenGLTransfer {
  GLenum type;
  int stride;
  GLuint pixelbuffers[SCHRO_OPENGL_TRANSFER_PIXELBUFFERS];
  int heights[SCHRO_OPENGL_TRANSFER_PIXELBUFFERS];
};

struct _SchroOpenGLCanvas {
  SchroOpenGL *opengl;
  SchroFrameFormat format;
  int width;
  int height;
  SchroOpenGLTexture texture;
  GLuint framebuffers[2];
  SchroOpenGLTransfer push;
  SchroOpenGLTransfer pull;
};

#define SCHRO_OPENGL_CANVAS_POOL_SIZE 50

// FIXME: add a mechanism to drop long time unused canvases from the pool
struct _SchroOpenGLCanvasPool {
  SchroOpenGL *opengl;
  SchroOpenGLCanvas *canvases[SCHRO_OPENGL_CANVAS_POOL_SIZE];
  int size;
};

// FIXME: reduce storage flags to fixed point, float and integer
#define SCHRO_OPENGL_CANVAS_STORE_BGRA           (1 <<  0)
#define SCHRO_OPENGL_CANVAS_STORE_U8_AS_UI8      (1 <<  1)
#define SCHRO_OPENGL_CANVAS_STORE_U8_AS_F16      (1 <<  2)
#define SCHRO_OPENGL_CANVAS_STORE_U8_AS_F32      (1 <<  3)
#define SCHRO_OPENGL_CANVAS_STORE_S16_AS_UI16    (1 <<  4)
#define SCHRO_OPENGL_CANVAS_STORE_S16_AS_I16     (1 <<  5)
#define SCHRO_OPENGL_CANVAS_STORE_S16_AS_U16     (1 <<  6)
#define SCHRO_OPENGL_CANVAS_STORE_S16_AS_F16     (1 <<  7)
#define SCHRO_OPENGL_CANVAS_STORE_S16_AS_F32     (1 <<  8)

#define SCHRO_OPENGL_CANVAS_PUSH_RENDER_QUAD     (1 <<  9)
#define SCHRO_OPENGL_CANVAS_PUSH_SHADER          (1 << 10)
#define SCHRO_OPENGL_CANVAS_PUSH_DRAWPIXELS      (1 << 11)
#define SCHRO_OPENGL_CANVAS_PUSH_U8_PIXELBUFFER  (1 << 12)
#define SCHRO_OPENGL_CANVAS_PUSH_U8_AS_F32       (1 << 13)
#define SCHRO_OPENGL_CANVAS_PUSH_S16_PIXELBUFFER (1 << 14)
#define SCHRO_OPENGL_CANVAS_PUSH_S16_AS_U16      (1 << 15)
#define SCHRO_OPENGL_CANVAS_PUSH_S16_AS_F32      (1 << 16)

#define SCHRO_OPENGL_CANVAS_PULL_PIXELBUFFER     (1 << 17)
#define SCHRO_OPENGL_CANVAS_PULL_U8_AS_F32       (1 << 18)
#define SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16      (1 << 19)
#define SCHRO_OPENGL_CANVAS_PULL_S16_AS_F32      (1 << 20)

extern unsigned int _schro_opengl_canvas_flags;

#define SCHRO_OPENGL_CANVAS_IS_FLAG_SET(_flag) \
    (_schro_opengl_canvas_flags & SCHRO_OPENGL_CANVAS_##_flag)
#define SCHRO_OPENGL_CANVAS_SET_FLAG(_flag) \
    (_schro_opengl_canvas_flags |= SCHRO_OPENGL_CANVAS_##_flag)
#define SCHRO_OPENGL_CANVAS_CLEAR_FLAG(_flag) \
    (_schro_opengl_canvas_flags &= ~SCHRO_OPENGL_CANVAS_##_flag)

void schro_opengl_canvas_check_flags (void);
void schro_opengl_canvas_print_flags (const char* indent);

SchroOpenGLCanvas *schro_opengl_canvas_new (SchroOpenGL *opengl,
    SchroFrameFormat format, int width, int height);
void schro_opengl_canvas_free (SchroOpenGLCanvas *canvas);
void schro_opengl_canvas_push (SchroOpenGLCanvas *dest,
    SchroFrameData *src); // CPU -> GPU
void schro_opengl_canvas_pull (SchroFrameData *dest,
    SchroOpenGLCanvas *src); // CPU <- GPU

SchroOpenGLCanvasPool *schro_opengl_canvas_pool_new (SchroOpenGL *opengl);
void schro_opengl_canvas_pool_free (SchroOpenGLCanvasPool* canvas_pool);
SchroOpenGLCanvas *schro_opengl_canvas_pool_pull_or_new (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGL *opengl, SchroFrameFormat format, int width, int height);
void schro_opengl_canvas_pool_push_or_free (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGLCanvas *canvas);

SCHRO_END_DECLS

#endif

