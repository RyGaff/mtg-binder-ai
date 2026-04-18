#import "CardDetectorBridge.h"
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>
#import <VisionCamera/FrameProcessorPlugin.h>
#import <VisionCamera/FrameProcessorPluginRegistry.h>
#import <VisionCamera/VisionCameraProxyHolder.h>
#import <VisionCamera/Frame.h>
#include "../cpp/card_detector.h"

// ── cv::Mat helpers ───────────────────────────────────────────────────────────

static cv::Mat matFromUIImage(UIImage *image) {
    CGImageRef cgImage = image.CGImage;
    size_t w = CGImageGetWidth(cgImage);
    size_t h = CGImageGetHeight(cgImage);
    cv::Mat rgba((int)h, (int)w, CV_8UC4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data, w, h, 8, rgba.step[0], cs,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
    );
    CGColorSpaceRelease(cs);
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cgImage);
    CGContextRelease(ctx);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

static cv::Mat grayMatFromSampleBuffer(CMSampleBufferRef sampleBuffer) {
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    void *base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    size_t w   = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
    size_t h   = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
    cv::Mat gray((int)h, (int)w, CV_8UC1, base);
    cv::Mat grayClone = gray.clone();
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return grayClone;
}

// ── Dict builder ─────────────────────────────────────────────────────────────

static NSDictionary<NSString *, id> *cornersToDict(
    const CardCorners& c,
    NSString * _Nullable rectifiedUri
) {
    NSMutableDictionary *d = [@{
        @"topLeftX":     @(c.topLeftX),
        @"topLeftY":     @(c.topLeftY),
        @"topRightX":    @(c.topRightX),
        @"topRightY":    @(c.topRightY),
        @"bottomRightX": @(c.bottomRightX),
        @"bottomRightY": @(c.bottomRightY),
        @"bottomLeftX":  @(c.bottomLeftX),
        @"bottomLeftY":  @(c.bottomLeftY),
        @"confidence":   @(c.confidence),
    } mutableCopy];
    if (rectifiedUri) d[@"rectifiedUri"] = rectifiedUri;
    return [d copy];
}

// ── ObjC frame processor plugin ──────────────────────────────────────────────

@interface CardDetectorFrameProcessorPlugin : FrameProcessorPlugin
@end

@implementation CardDetectorFrameProcessorPlugin

- (instancetype)initWithProxy:(VisionCameraProxyHolder*)proxy withOptions:(NSDictionary*)options {
    return [super initWithProxy:proxy withOptions:options];
}

- (id)callback:(Frame*)frame withArguments:(NSDictionary*)arguments {
    cv::Mat gray = grayMatFromSampleBuffer(frame.buffer);
    if (gray.empty()) return nil;
    CardCorners corners;
    if (!detectCardCorners(gray, corners, nullptr)) return nil;
    return cornersToDict(corners, nil);
}

@end

// ── Registration ─────────────────────────────────────────────────────────────

@implementation CardDetectorBridge (PluginRegistration)

+ (void)registerFrameProcessorPlugin {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSLog(@"[CardDetector] Registering detectCardCornersInFrame...");
        [FrameProcessorPluginRegistry
            addFrameProcessorPlugin:@"detectCardCornersInFrame"
                    withInitializer:^FrameProcessorPlugin*(VisionCameraProxyHolder* proxy, NSDictionary* options) {
            return [[CardDetectorFrameProcessorPlugin alloc] initWithProxy:proxy withOptions:options];
        }];
        NSLog(@"[CardDetector] Plugin registered OK");
    });
}

@end

// ── Bridge methods ────────────────────────────────────────────────────────────

@implementation CardDetectorBridge

+ (void)load {
    NSLog(@"[CardDetector] +load fired — registering plugin");
    [self registerFrameProcessorPlugin];
}

+ (nullable NSDictionary<NSString *, id> *)detectCornersFromFileURI:(NSString *)uri {
    NSString *path = [uri hasPrefix:@"file://"] ? [uri substringFromIndex:7] : uri;
    UIImage *uiImage = [UIImage imageWithContentsOfFile:path];
    if (!uiImage || !uiImage.CGImage) return nil;

    cv::Mat bgr = matFromUIImage(uiImage);
    if (bgr.empty()) return nil;

    NSString *rectPath = [path stringByAppendingString:@".rect.jpg"];
    std::string rectPathStr = rectPath.UTF8String;

    CardCorners corners;
    if (!detectCardCorners(bgr, corners, &rectPathStr)) return nil;

    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *rectUri = [fm fileExistsAtPath:rectPath]
        ? [@"file://" stringByAppendingString:rectPath]
        : nil;

    return cornersToDict(corners, rectUri);
}

+ (nullable NSDictionary<NSString *, id> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer {
    cv::Mat gray = grayMatFromSampleBuffer(sampleBuffer);
    if (gray.empty()) return nil;
    CardCorners corners;
    if (!detectCardCorners(gray, corners, nullptr)) return nil;
    return cornersToDict(corners, nil);
}

@end
