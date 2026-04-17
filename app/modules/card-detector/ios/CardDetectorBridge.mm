#import "CardDetectorBridge.h"
#import <Vision/Vision.h>
#import <UIKit/UIKit.h>

static NSDictionary<NSString *, NSNumber *> *observationToDict(VNRectangleObservation *obs) {
    // Vision uses bottom-left origin — flip Y to top-left for JS
    return @{
        @"topLeftX":     @(obs.topLeft.x),
        @"topLeftY":     @(1.0 - obs.topLeft.y),
        @"topRightX":    @(obs.topRight.x),
        @"topRightY":    @(1.0 - obs.topRight.y),
        @"bottomRightX": @(obs.bottomRight.x),
        @"bottomRightY": @(1.0 - obs.bottomRight.y),
        @"bottomLeftX":  @(obs.bottomLeft.x),
        @"bottomLeftY":  @(1.0 - obs.bottomLeft.y),
    };
}

static VNDetectRectanglesRequest *makeRectangleRequest(void (^handler)(VNRequest *, NSError *)) {
    VNDetectRectanglesRequest *req = [[VNDetectRectanglesRequest alloc] initWithCompletionHandler:handler];
    req.minimumAspectRatio  = 0.62f;
    req.maximumAspectRatio  = 0.78f;
    req.minimumSize         = 0.10f;
    req.maximumObservations = 1;
    return req;
}

@implementation CardDetectorBridge

+ (nullable NSDictionary<NSString *, NSNumber *> *)detectCornersFromFileURI:(NSString *)uri {
    NSString *path = [uri hasPrefix:@"file://"] ? [uri substringFromIndex:7] : uri;
    UIImage *uiImage = [UIImage imageWithContentsOfFile:path];
    if (!uiImage || !uiImage.CGImage) return nil;

    __block NSDictionary<NSString *, NSNumber *> *result = nil;
    VNDetectRectanglesRequest *request = makeRectangleRequest(^(VNRequest *req, NSError *err) {
        VNRectangleObservation *obs = req.results.firstObject;
        if (obs) result = observationToDict(obs);
    });

    CIImage *ciImage = [CIImage imageWithCGImage:uiImage.CGImage];
    VNImageRequestHandler *imgHandler = [[VNImageRequestHandler alloc] initWithCIImage:ciImage options:@{}];
    [imgHandler performRequests:@[request] error:nil];
    return result;
}

+ (nullable NSDictionary<NSString *, NSNumber *> *)detectCornersFromSampleBuffer:(CMSampleBufferRef)sampleBuffer {
    __block NSDictionary<NSString *, NSNumber *> *result = nil;
    VNDetectRectanglesRequest *request = makeRectangleRequest(^(VNRequest *req, NSError *err) {
        VNRectangleObservation *obs = req.results.firstObject;
        if (obs) result = observationToDict(obs);
    });

    // initWithCMSampleBuffer:options: available iOS 14+ (our min is 15.1)
    VNImageRequestHandler *imgHandler = [[VNImageRequestHandler alloc] initWithCMSampleBuffer:sampleBuffer options:@{}];
    [imgHandler performRequests:@[request] error:nil];
    return result;
}

@end
