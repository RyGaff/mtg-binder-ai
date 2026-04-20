#import "CardDetectorBridge.h"
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreML/CoreML.h>
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

    // takePhoto() and library images carry EXIF orientation on iOS.
    // CGImage returns the *unoriented* sensor bitmap, so for portrait photos
    // the Mat is landscape unless we apply the orientation manually.
    switch (image.imageOrientation) {
        case UIImageOrientationRight: {
            cv::Mat r;
            cv::rotate(bgr, r, cv::ROTATE_90_CLOCKWISE);
            bgr = r;
            break;
        }
        case UIImageOrientationLeft: {
            cv::Mat r;
            cv::rotate(bgr, r, cv::ROTATE_90_COUNTERCLOCKWISE);
            bgr = r;
            break;
        }
        case UIImageOrientationDown: {
            cv::Mat r;
            cv::rotate(bgr, r, cv::ROTATE_180);
            bgr = r;
            break;
        }
        case UIImageOrientationUp:
        default:
            break;
    }
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
    // Handle any combination of "file://" / "file:///" prefixes VisionCamera
    // or other callers might hand us.
    NSString *path = uri;
    if ([path hasPrefix:@"file://"]) {
        NSURL *url = [NSURL URLWithString:path];
        if (url && url.isFileURL) {
            path = url.path;
        } else {
            path = [path substringFromIndex:7];
        }
    }
    BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:path];
    NSLog(@"[CardDetector] photo uri=%@ path=%@ exists=%d", uri, path, exists);

    UIImage *uiImage = [UIImage imageWithContentsOfFile:path];
    if (!uiImage || !uiImage.CGImage) {
        return @{
            @"_error": @"uiimage_load_failed",
            @"resolvedPath": path,
            @"fileExists":   @(exists),
        };
    }
    NSLog(@"[CardDetector] photo size=%.0fx%.0f orient=%ld",
          uiImage.size.width, uiImage.size.height, (long)uiImage.imageOrientation);

    cv::Mat bgr = matFromUIImage(uiImage);
    if (bgr.empty()) {
        return @{ @"_error": @"cvmat_convert_failed" };
    }

    // Defensive: if EXIF orientation was absent/baked-in-place and the Mat
    // is still landscape, rotate to portrait.
    if (bgr.cols > bgr.rows) {
        cv::Mat r;
        cv::rotate(bgr, r, cv::ROTATE_90_CLOCKWISE);
        bgr = r;
        NSLog(@"[CardDetector] photo fallback-rotated to %dx%d", bgr.cols, bgr.rows);
    }

    // Downscale very large photos (takePhoto can yield 4032×3024). Keeping the
    // long edge ~1920 makes the 3% minArea threshold meaningful for a normal-sized
    // card and runs ~10x faster.
    double longEdge = std::max((double)bgr.cols, (double)bgr.rows);
    cv::Mat bgrSmall;
    if (longEdge > 1920.0) {
        double scale = 1920.0 / longEdge;
        cv::resize(bgr, bgrSmall, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        bgrSmall = bgr;
    }
    NSLog(@"[CardDetector] photo processed mat=%dx%d", bgrSmall.cols, bgrSmall.rows);

    NSString *rectPath = [path stringByAppendingString:@".rect.jpg"];
    std::string rectPathStr = rectPath.UTF8String;

    CardCorners corners;
    DetectionStats stats;
    if (!detectCardCorners(bgrSmall, corners, &rectPathStr, &stats)) {
        NSLog(@"[CardDetector] photo detect FAIL: contours=%d 4vert=%d area=%d conv=%d ang=%d AR=%d",
              stats.contoursTotal, stats.passed4Vertex, stats.passedMinArea,
              stats.passedConvex, stats.passedAngles, stats.passedAR);
        return @{
            @"_error": @"detect_failed",
            @"stats": @{
                @"contoursTotal": @(stats.contoursTotal),
                @"passed4Vertex": @(stats.passed4Vertex),
                @"passedMinArea": @(stats.passedMinArea),
                @"passedConvex":  @(stats.passedConvex),
                @"passedAngles":  @(stats.passedAngles),
                @"passedAR":      @(stats.passedAR),
            },
        };
    }

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

// ── CoreML image encoder ─────────────────────────────────────────────────────

+ (nullable MLModel *)loadEncoderModel {
    static MLModel *model = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSString *pathC = [[NSBundle mainBundle] pathForResource:@"card_encoder" ofType:@"mlmodelc"];
        NSString *pathRaw = [[NSBundle mainBundle] pathForResource:@"card_encoder" ofType:@"mlmodel"];
        NSLog(@"[CardDetector] encoder lookup — mlmodelc=%@ mlmodel=%@", pathC ?: @"(nil)", pathRaw ?: @"(nil)");

        NSString *path = pathC;
        NSURL *url = nil;
        if (path) {
            url = [NSURL fileURLWithPath:path];
        } else if (pathRaw) {
            NSError *compileErr = nil;
            NSURL *rawUrl = [NSURL fileURLWithPath:pathRaw];
            url = [MLModel compileModelAtURL:rawUrl error:&compileErr];
            if (compileErr || !url) {
                NSLog(@"[CardDetector] mlmodel compile failed: %@", compileErr);
                return;
            }
            NSLog(@"[CardDetector] compiled .mlmodel at runtime → %@", url.path);
        } else {
            NSLog(@"[CardDetector] card_encoder not bundled (neither .mlmodelc nor .mlmodel)");
            return;
        }
        NSError *err = nil;
        model = [MLModel modelWithContentsOfURL:url error:&err];
        if (err) {
            NSLog(@"[CardDetector] failed to load encoder: %@", err);
            model = nil;
        } else {
            NSLog(@"[CardDetector] encoder loaded ok");
        }
    });
    return model;
}

+ (nonnull NSDictionary<NSString *, id> *)encodeImageFromFileURI:(NSString *)uri {
    MLModel *model = [self loadEncoderModel];
    if (!model) return @{@"error": @"model not loaded (see launch-time CardDetector logs)"};

    NSString *path = [uri hasPrefix:@"file://"] ? [uri substringFromIndex:7] : uri;
    UIImage *ui = [UIImage imageWithContentsOfFile:path];
    if (!ui || !ui.CGImage) return @{@"error": [NSString stringWithFormat:@"image decode failed: %@", path]};

    // Resize to 224×224 and render to a BGRA pixel buffer.
    CGSize target = CGSizeMake(224, 224);
    UIGraphicsBeginImageContextWithOptions(target, YES, 1.0);
    [ui drawInRect:CGRectMake(0, 0, target.width, target.height)];
    UIImage *resized = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    if (!resized || !resized.CGImage) return @{@"error": @"resize failed"};

    CVPixelBufferRef pb = NULL;
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
    };
    CVReturn st = CVPixelBufferCreate(
        kCFAllocatorDefault, 224, 224,
        kCVPixelFormatType_32ARGB,
        (__bridge CFDictionaryRef)attrs, &pb);
    if (st != kCVReturnSuccess || !pb) return @{@"error": [NSString stringWithFormat:@"CVPixelBufferCreate=%d", st]};

    CVPixelBufferLockBaseAddress(pb, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        CVPixelBufferGetBaseAddress(pb), 224, 224, 8,
        CVPixelBufferGetBytesPerRow(pb), cs,
        kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    CGContextDrawImage(ctx, CGRectMake(0, 0, 224, 224), resized.CGImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CVPixelBufferUnlockBaseAddress(pb, 0);

    NSError *err = nil;
    MLFeatureValue *fv = [MLFeatureValue featureValueWithPixelBuffer:pb];
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"input": fv} error:&err];
    CVPixelBufferRelease(pb);
    if (err || !input) return @{@"error": [NSString stringWithFormat:@"input build: %@", err.localizedDescription ?: @"unknown"]};

    id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&err];
    if (err || !output) return @{@"error": [NSString stringWithFormat:@"prediction: %@", err.localizedDescription ?: @"unknown"]};

    NSSet<NSString *> *names = [output featureNames];
    MLFeatureValue *vec = [output featureValueForName:@"output"];
    if (!vec) return @{@"error": [NSString stringWithFormat:@"no 'output' feature, available=%@", names]};

    MLMultiArray *arr = vec.multiArrayValue;
    if (!arr) return @{@"error": [NSString stringWithFormat:@"multiArrayValue nil type=%ld", (long)vec.type]};
    if (arr.dataType != MLMultiArrayDataTypeFloat32) {
        return @{@"error": [NSString stringWithFormat:@"unexpected dataType=%ld (want Float32=%ld)",
                            (long)arr.dataType, (long)MLMultiArrayDataTypeFloat32]};
    }

    NSUInteger count = arr.count;
    const float *src = (const float *)arr.dataPointer;
    NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:count];
    for (NSUInteger i = 0; i < count; i++) {
        [out addObject:@(src[i])];
    }
    return @{@"embedding": [out copy]};
}

@end
