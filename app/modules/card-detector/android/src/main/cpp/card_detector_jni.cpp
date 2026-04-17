#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "card_detector.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_expo_modules_carddetector_CardDetectorModule_detectCornersNative(
        JNIEnv *env, jobject /* this */, jstring filePath) {

    const char *path = env->GetStringUTFChars(filePath, nullptr);
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    env->ReleaseStringUTFChars(filePath, path);

    if (image.empty()) return nullptr;

    CardCorners corners;
    if (!detectCardCorners(image, corners)) return nullptr;

    jfloatArray result = env->NewFloatArray(8);
    float data[8] = {
        corners.topLeftX,     corners.topLeftY,
        corners.topRightX,    corners.topRightY,
        corners.bottomRightX, corners.bottomRightY,
        corners.bottomLeftX,  corners.bottomLeftY,
    };
    env->SetFloatArrayRegion(result, 0, 8, data);
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
    if (!detectCardCorners(grayClone, corners)) return nullptr;

    jfloatArray result = env->NewFloatArray(8);
    float data[8] = {
        corners.topLeftX,     corners.topLeftY,
        corners.topRightX,    corners.topRightY,
        corners.bottomRightX, corners.bottomRightY,
        corners.bottomLeftX,  corners.bottomLeftY,
    };
    env->SetFloatArrayRegion(result, 0, 8, data);
    return result;
}
