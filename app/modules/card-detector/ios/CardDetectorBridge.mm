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
    if (!pixelBuffer) return cv::Mat();

    OSType fmt = CVPixelBufferGetPixelFormatType(pixelBuffer);

    // One-time log so we can see the format in device console
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        char fcc[5] = {
            (char)((fmt >> 24) & 0xff), (char)((fmt >> 16) & 0xff),
            (char)((fmt >> 8)  & 0xff), (char)( fmt        & 0xff), 0
        };
        NSLog(@"[CardDetector] first frame pixelFormat=%s (%u) planar=%s w=%zu h=%zu",
              fcc, (unsigned)fmt,
              CVPixelBufferIsPlanar(pixelBuffer) ? "Y" : "N",
              CVPixelBufferGetWidth(pixelBuffer),
              CVPixelBufferGetHeight(pixelBuffer));
    });

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    cv::Mat result;

    if (CVPixelBufferIsPlanar(pixelBuffer)) {
        // YUV biplanar (420YpCbCr8Bi*) — plane 0 is luma
        void *base    = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
        size_t w      = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
        size_t h      = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
        size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
        if (base) {
            cv::Mat gray((int)h, (int)w, CV_8UC1, base, stride);
            result = gray.clone();
        }
    } else {
        // Non-planar — likely BGRA or RGBA. Convert to grayscale.
        void *base    = CVPixelBufferGetBaseAddress(pixelBuffer);
        size_t w      = CVPixelBufferGetWidth(pixelBuffer);
        size_t h      = CVPixelBufferGetHeight(pixelBuffer);
        size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);
        if (base) {
            cv::Mat bgra((int)h, (int)w, CV_8UC4, base, stride);
            cv::cvtColor(bgra, result, cv::COLOR_BGRA2GRAY);
        }
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return result;
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
    CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(frame.buffer);
    OSType fmt = pb ? CVPixelBufferGetPixelFormatType(pb) : 0;
    char fcc[5] = {
        (char)((fmt >> 24) & 0xff), (char)((fmt >> 16) & 0xff),
        (char)((fmt >> 8)  & 0xff), (char)( fmt        & 0xff), 0
    };
    NSString *fccStr = [NSString stringWithUTF8String:fcc];

    cv::Mat gray = grayMatFromSampleBuffer(frame.buffer);
    // Rotate landscape sensor buffer to match portrait display so corners come
    // out in screen-aligned coordinates. CCW matches iOS rear camera default
    // orientation when device is held portrait.
    if (!gray.empty() && gray.cols > gray.rows) {
        cv::Mat rotated;
        cv::rotate(gray, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        gray = rotated;
    }
    if (gray.empty()) {
        return @{ @"_debug": @YES, @"pixelFormat": fccStr,
                  @"frameW": @(0), @"frameH": @(0) };
    }

    CardCorners corners;
    DetectionStats stats;
    bool detected = detectCardCorners(gray, corners, nullptr, &stats);

    NSDictionary *statsDict = @{
        @"medianLuma":     @(stats.medianLuma),
        @"edgePixels":     @(stats.edgePixels),
        @"contoursTotal":  @(stats.contoursTotal),
        @"passed4Vertex":  @(stats.passed4Vertex),
        @"passedMinArea":  @(stats.passedMinArea),
        @"passedConvex":   @(stats.passedConvex),
        @"passedAngles":   @(stats.passedAngles),
        @"passedAR":       @(stats.passedAR),
    };

    if (!detected) {
        return @{ @"_debug":      @YES,
                  @"pixelFormat": fccStr,
                  @"frameW":      @(gray.cols),
                  @"frameH":      @(gray.rows),
                  @"stats":       statsDict };
    }

    NSMutableDictionary *d = [cornersToDict(corners, nil) mutableCopy];
    d[@"pixelFormat"] = fccStr;
    d[@"frameW"]      = @(gray.cols);
    d[@"frameH"]      = @(gray.rows);
    d[@"stats"]       = statsDict;
    return [d copy];
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
