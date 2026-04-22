#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "card_detector.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_expo_modules_carddetector_CardDetectorModule_detectCornersNative(
        JNIEnv *env, jobject /* this */, jstring filePath) {

    const char *path = env->GetStringUTFChars(filePath, nullptr);
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    std::string rectPath = std::string(path) + ".rect.jpg";
    env->ReleaseStringUTFChars(filePath, path);

    if (image.empty()) return nullptr;

    CardCorners corners;
    if (!detectCardCorners(image, corners, &rectPath)) return nullptr;

    jfloatArray result = env->NewFloatArray(10);
    if (!result) return nullptr;
    float data[10] = {
        corners.topLeftX,     corners.topLeftY,
        corners.topRightX,    corners.topRightY,
        corners.bottomRightX, corners.bottomRightY,
        corners.bottomLeftX,  corners.bottomLeftY,
        corners.confidence,
        (float)corners.source,
    };
    env->SetFloatArrayRegion(result, 0, 10, data);
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_expo_modules_carddetector_CardDetectorFrameProcessorPlugin_detectCornersFromGrayscaleNative(
        JNIEnv *env, jobject /* this */, jbyteArray grayscaleBytes, jint width, jint height) {

    jbyte* bytes = env->GetByteArrayElements(grayscaleBytes, nullptr);
    cv::Mat gray(height, width, CV_8UC1, reinterpret_cast<uint8_t*>(bytes));
    cv::Mat grayClone = gray.clone();
    env->ReleaseByteArrayElements(grayscaleBytes, bytes, JNI_ABORT);

    CardCorners corners;
    // No rectified image for frame processor path — OCR uses high-res photo
    if (!detectCardCorners(grayClone, corners, nullptr)) return nullptr;

    jfloatArray result = env->NewFloatArray(10);
    if (!result) return nullptr;
    float data[10] = {
        corners.topLeftX,     corners.topLeftY,
        corners.topRightX,    corners.topRightY,
        corners.bottomRightX, corners.bottomRightY,
        corners.bottomLeftX,  corners.bottomLeftY,
        corners.confidence,
        (float)corners.source,
    };
    env->SetFloatArrayRegion(result, 0, 10, data);
    return result;
}
