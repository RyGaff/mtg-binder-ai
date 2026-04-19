#pragma once
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>

@interface CardDetectorBridge : NSObject
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromFileURI:(NSString *)uri;
+ (nullable NSDictionary<NSString *, id> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
+ (void)registerFrameProcessorPlugin;

/**
 * Run the bundled CoreML image encoder on the file at `uri`. Returns a
 * 256-length NSArray of NSNumber (float) or nil when the encoder asset
 * (card_encoder.mlmodelc) is not present in the bundle.
 */
+ (nullable NSArray<NSNumber *> *)encodeImageFromFileURI:(NSString *)uri;
@end
