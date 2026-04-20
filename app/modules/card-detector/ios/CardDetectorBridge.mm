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
        NSString *pathC   = [[NSBundle mainBundle] pathForResource:@"card_encoder" ofType:@"mlmodelc"];
        NSString *pathRaw = [[NSBundle mainBundle] pathForResource:@"card_encoder" ofType:@"mlmodel"];
        NSURL *url = nil;
        if (pathC) {
            url = [NSURL fileURLWithPath:pathC];
        } else if (pathRaw) {
            // CocoaPods' s.resources copies .mlmodel as-is; compile at runtime
            // if Xcode didn't produce a .mlmodelc.
            NSError *compileErr = nil;
            url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pathRaw] error:&compileErr];
            if (compileErr || !url) {
                NSLog(@"[CardDetector] mlmodel compile failed: %@", compileErr);
                return;
            }
        } else {
            NSLog(@"[CardDetector] card_encoder not bundled — encodeImage disabled");
            return;
        }
        NSError *err = nil;
        model = [MLModel modelWithContentsOfURL:url error:&err];
        if (err) {
            NSLog(@"[CardDetector] failed to load encoder: %@", err);
            model = nil;
        }
    });
    return model;
}

+ (nonnull NSDictionary<NSString *, id> *)encodeImageFromFileURI:(NSString *)uri {
    // Mirrors the Python training-time eval transform step-for-step:
    //   1. decode file → UIImage  (RGB)
    //   2. LongestMaxSize(224)    — aspect-preserving scale, long side = 224
    //   3. PadIfNeeded(224,224)   — center-pad with zeros
    //   4. extract RGB bytes, normalize to [0, 1] float32
    //   5. fill NCHW tensor [1,3,224,224]  (TFLite side uses NHWC but
    //      same steps 1-4; the channel-order difference is dictated by the
    //      respective model exports)
    //   6. run inference
    //   7. copy 256-float output to an NSArray

    MLModel *model = [self loadEncoderModel];
    if (!model) return @{@"error": @"model not loaded (see launch-time CardDetector logs)"};

    NSString *path = [uri hasPrefix:@"file://"] ? [uri substringFromIndex:7] : uri;
    UIImage *ui = [UIImage imageWithContentsOfFile:path];
    if (!ui || !ui.CGImage) return @{@"error": [NSString stringWithFormat:@"image decode failed: %@", path]};

    // Compute aspect-preserving target rect inside a 224×224 canvas.
    const NSUInteger side = 224;
    const NSUInteger plane = side * side;
    const CGFloat origW = CGImageGetWidth(ui.CGImage);
    const CGFloat origH = CGImageGetHeight(ui.CGImage);
    const CGFloat scale = (CGFloat)side / MAX(origW, origH);
    const CGFloat drawW = origW * scale;
    const CGFloat drawH = origH * scale;
    const CGFloat offsetX = ((CGFloat)side - drawW) / 2.0;
    const CGFloat offsetY = ((CGFloat)side - drawH) / 2.0;

    // calloc zero-fills, so pixels outside the drawn rect are black — matches
    // albumentations PadIfNeeded(border_mode=BORDER_CONSTANT, fill=0).
    uint8_t *raw = (uint8_t *)calloc(plane * 4, sizeof(uint8_t));
    if (!raw) return @{@"error": @"calloc failed"};

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        raw, side, side, 8, side * 4, cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctx) { free(raw); return @{@"error": @"CGBitmapContextCreate failed"}; }
    // Flip Y so our top-down row-major raw buffer matches the orientation
    // PIL / BitmapFactory produce (Core Graphics has y=0 at the bottom,
    // so CGContextDrawImage without this flip writes the image upside-down).
    CGContextTranslateCTM(ctx, 0, side);
    CGContextScaleCTM(ctx, 1.0, -1.0);
    CGContextDrawImage(ctx, CGRectMake(offsetX, offsetY, drawW, drawH), ui.CGImage);
    CGContextRelease(ctx);

    // Model signature (from coremlcompiler metadata):
    //   input  "image"     MultiArray Float32 [1, 3, 224, 224]  (NCHW)
    //   output "embedding" MultiArray Float32 [1, 256]
    NSError *err = nil;
    MLMultiArray *mla = [[MLMultiArray alloc]
        initWithShape:@[@1, @3, @(side), @(side)]
             dataType:MLMultiArrayDataTypeFloat32
                error:&err];
    if (err || !mla) { free(raw); return @{@"error": [NSString stringWithFormat:@"MLMultiArray init: %@", err.localizedDescription ?: @"unknown"]}; }

    // NCHW: fill all red values, then all green, then all blue. Each plane
    // is 224*224 floats.
    float *dst = (float *)mla.dataPointer;
    for (NSUInteger i = 0; i < plane; i++) {
        const NSUInteger src = i * 4;  // RGBA
        dst[0 * plane + i] = raw[src + 0] / 255.0f;  // R
        dst[1 * plane + i] = raw[src + 1] / 255.0f;  // G
        dst[2 * plane + i] = raw[src + 2] / 255.0f;  // B
    }
    free(raw);

    MLFeatureValue *fv = [MLFeatureValue featureValueWithMultiArray:mla];
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"image": fv} error:&err];
    if (err || !input) return @{@"error": [NSString stringWithFormat:@"input build: %@", err.localizedDescription ?: @"unknown"]};

    id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&err];
    if (err || !output) return @{@"error": [NSString stringWithFormat:@"prediction: %@", err.localizedDescription ?: @"unknown"]};

    MLFeatureValue *vec = [output featureValueForName:@"embedding"];
    if (!vec) return @{@"error": [NSString stringWithFormat:@"no 'embedding' feature, available=%@", [output featureNames]]};

    MLMultiArray *arr = vec.multiArrayValue;
    if (!arr) return @{@"error": [NSString stringWithFormat:@"multiArrayValue nil type=%ld", (long)vec.type]};
    if (arr.dataType != MLMultiArrayDataTypeFloat32) {
        return @{@"error": [NSString stringWithFormat:@"unexpected output dataType=%ld (want Float32=%ld)",
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
