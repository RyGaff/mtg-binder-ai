#pragma once
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>

@interface CardDetectorBridge : NSObject
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromFileURI:(NSString *)uri;
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
+ (void)registerFrameProcessorPlugin;

/**
 * Run the bundled CoreML image encoder on the file at `uri`. Returns a
 * dictionary with one of:
 *   { "embedding": NSArray<NSNumber> }  on success (256-length float32 vec)
 *   { "error":     NSString }           on any failure with a reason
 */
+ (nonnull NSDictionary<NSString *, id> *)encodeImageFromFileURI:(NSString *)uri;
@end
