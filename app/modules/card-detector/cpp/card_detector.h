#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#define CARD_CONFIDENCE_MIN    0.35f
#define CARD_CONFIDENCE_STABLE 0.65f

struct CardCorners {
    float topLeftX,     topLeftY;
    float topRightX,    topRightY;
    float bottomRightX, bottomRightY;
    float bottomLeftX,  bottomLeftY;
    float confidence;
};

/**
 * Detects the card-shaped quad with highest confidence in the image.
 * Corners normalized to 0–1 (top-left origin). Returns false if no quad found.
 * If rectifiedPath is non-null and non-empty, writes a 400×560 perspective-corrected
 * JPEG to that path before returning.
 */
bool detectCardCorners(const cv::Mat& image, CardCorners& out,
                       std::string* rectifiedPath = nullptr);
