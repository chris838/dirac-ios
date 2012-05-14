//  Created by Chris Harding on 06/05/2012.
//  Copyright (c) 2012 Swift Navigation. All rights reserved.
//

#import "ViewController.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>

#import "test-schro.c"

static void
frame_free (SchroFrame *frame, void *priv)
{
    free (priv);
}

static void
test (int w, int h)
{
    int size;
    uint8_t *picture;
    SchroEncoder *encoder;
    SchroBuffer *buffer;
    SchroFrame *frame;
    SchroVideoFormat *format;
    int n_frames;
    int go;
    
    encoder = schro_encoder_new();
    format = schro_encoder_get_video_format(encoder);
    format->width = w;
    format->height = h;
    format->clean_width = w;
    format->clean_height = h;
    format->left_offset = 0;
    format->top_offset = 0;
    schro_encoder_set_video_format (encoder, format);
    free (format);
    schro_encoder_start (encoder);
    
    size = ROUND_UP_4 (w) * ROUND_UP_2 (h);
    size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);
    size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);
    
    n_frames = 0;
    go = 1;
    while (go) {
        int x;
        
        switch (schro_encoder_wait (encoder)) {
            case SCHRO_STATE_NEED_FRAME:
                if (n_frames < 100) {
                    //SCHRO_ERROR("frame %d", n_frames);
                    
                    picture = (uint8_t *) malloc(size);
                    memset (picture, 128, size);
                    
                    frame = schro_frame_new_from_data_I420 (picture, w, h);
                    
                    schro_frame_set_free_callback (frame, frame_free, picture);
                    
                    schro_encoder_push_frame (encoder, frame);
                    
                    NSLog(@"Pushed frame %u", n_frames);
                    
                    n_frames++;
                } else {
                    schro_encoder_end_of_stream (encoder);
                }
                break;
            case SCHRO_STATE_HAVE_BUFFER:
                buffer = schro_encoder_pull (encoder, &x);
                printf("%d\n", x);
                schro_buffer_unref (buffer);
                break;
            case SCHRO_STATE_AGAIN:
                break;
            case SCHRO_STATE_END_OF_STREAM:
                go = 0;
                break;
            default:
                break;
        }
    }
    
    schro_encoder_free (encoder);
}


@implementation ViewController


#define SIZE 128

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    NSLog(@"Testing schro library via orc");
    run_schro_test();
    
    
    NSLog(@"Testing schro library encode");
    int h, w;
    
    schro_init();
    
    test(853,480);
    if (0) {
        for(w=SIZE;w<SIZE+16;w++){
            for(h=SIZE;h<SIZE+16;h++){
                test(w,h);
            }
        }
    }
    
}

- (void)viewDidUnload
{
    [super viewDidUnload];
    // Release any retained subviews of the main view.
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone) {
        return (interfaceOrientation != UIInterfaceOrientationPortraitUpsideDown);
    } else {
        return YES;
    }
}

@end
