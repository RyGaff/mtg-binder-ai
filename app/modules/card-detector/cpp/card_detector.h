#pragma once
#include <opencv2/opencv.hpp>

struct CardCorners {
    float topLeftX,     topLeftY;
    float topRightX,    topRightY;
    float bottomRightX, bottomRightY;
    float bottomLeftX,  bottomLeftY;
};

/**
 * Detects the largest card-shaped rectangle in the image.
 * Corners are normalized to 0–1 range (top-left origin).
 * Returns false if no qualifying 4-vertex contour found.
 */
bool detectCardCorners(const cv::Mat& image, CardCorners& out);
