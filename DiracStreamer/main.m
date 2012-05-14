//
//  main.m
//  DiracStreamer
//
//  Created by Chris Harding on 06/05/2012.
//  Copyright (c) 2012 Swift Navigation. All rights reserved.
//

#import <UIKit/UIKit.h>

#import "AppDelegate.h"

int main(int argc, char *argv[])
{
    int retVal = 0;
    
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    //@autoreleasepool {
    
        retVal = UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    
    //}
    [pool drain];
    
    return retVal;
}
