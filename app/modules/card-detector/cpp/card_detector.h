#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#define CARD_CONFIDENCE_MIN    0.35f
#define CARD_CONFIDENCE_STABLE 0.65f

// Which detection path produced this result. Callers use `source` to
// pick per-source rectification expansion — line-pair quads are inset
// from the card outer edge, Otsu quads include a small background
// halo, primary quads fit the card exactly.
enum CardSource {
    CARD_SOURCE_PRIMARY    = 0,
    CARD_SOURCE_LINEINTERP = 1,
    CARD_SOURCE_OTSU       = 2,
};

struct CardCorners {
    float topLeftX,     topLeftY;
    float topRightX,    topRightY;
    float bottomRightX, bottomRightY;
    float bottomLeftX,  bottomLeftY;
    float confidence;
    int   source;   // CardSource enum value
};

// Per-stage counters so we can see where the pipeline is rejecting quads.
struct DetectionStats {
    int medianLuma;     // median gray value feeding the adaptive Canny
    int edgePixels;     // non-zero pixels in the post-Canny+dilate edge map
    int contoursTotal;
    int passed4Vertex;
    int passedMinArea;
    int passedConvex;
    int passedAngles;
    int passedAR;
};

/**
 * Detects the card-shaped quad with highest confidence in the image.
 * Corners normalized to 0–1 (top-left origin). Returns false if no quad found.
 * If rectifiedPath is non-null and non-empty, writes a 400×560 perspective-corrected
 * JPEG to that path before returning.
 * If stats is non-null, it is filled with per-stage counters for diagnostics.
 */
bool detectCardCorners(const cv::Mat& image, CardCorners& out,
                       std::string* rectifiedPath = nullptr,
                       DetectionStats* stats = nullptr);
