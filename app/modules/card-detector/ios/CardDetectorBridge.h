#pragma once
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>

@interface CardDetectorBridge : NSObject
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromFileURI:(NSString *)uri;
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
+ (void)registerFrameProcessorPlugin;
@end
