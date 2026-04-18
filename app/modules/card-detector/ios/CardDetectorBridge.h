#pragma once
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>

@interface CardDetectorBridge : NSObject
+ (nullable NSDictionary<NSString *, NSNumber *> *)detectCornersFromFileURI:(NSString *)uri;
+ (nullable NSDictionary<NSString *, NSNumber *> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
+ (void)registerFrameProcessorPlugin;
@end
