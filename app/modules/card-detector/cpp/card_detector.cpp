#include "card_detector.h"
#include <algorithm>
#include <vector>

// Histogram-based O(n) median — faster than sorting for large images.
static double imageMedian(const cv::Mat& gray) {
    int histSize = 256;
    float range[] = {0.0f, 256.0f};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);
    int total = gray.rows * gray.cols;
    int cumSum = 0;
    for (int i = 0; i < 256; i++) {
        cumSum += static_cast<int>(hist.at<float>(i));
        if (cumSum >= total / 2) return static_cast<double>(i);
    }
    return 128.0;
}

bool detectCardCorners(const cv::Mat& image, CardCorners& out) {
    if (image.empty()) return false;

    // ── 1. Grayscale ─────────────────────────────────────────────────────────
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    // ── 2. CLAHE — normalize contrast across lighting conditions ─────────────
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat equalized;
    clahe->apply(gray, equalized);

    // ── 3. Gaussian blur ─────────────────────────────────────────────────────
    cv::Mat blurred;
    cv::GaussianBlur(equalized, blurred, cv::Size(5, 5), 0);

    // ── 4. Adaptive Canny — thresholds derived from image median ─────────────
    double median = imageMedian(blurred);
    double lo = std::max(0.0,   0.67 * median);
    double hi = std::min(255.0, 1.33 * median);
    cv::Mat edges;
    cv::Canny(blurred, edges, lo, hi);

    // ── 5. Dilate — close small gaps in card edges ───────────────────────────
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 1);

    // ── 6. Contour detection ─────────────────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double imageArea = static_cast<double>(image.cols) * image.rows;
    double minArea = imageArea * 0.10;

    std::vector<cv::Point> best;
    double bestArea = 0;

    for (const auto& contour : contours) {
        double perimeter = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        if (approx.size() != 4) continue;

        double area = cv::contourArea(approx);
        if (area < minArea) continue;

        // Validate aspect ratio matches an MTG card (portrait or landscape)
        cv::Rect bounds = cv::boundingRect(approx);
        if (bounds.height == 0) continue;
        double ratio = static_cast<double>(bounds.width) / bounds.height;
        bool portrait  = ratio >= 0.55 && ratio <= 0.80;
        bool landscape = ratio >= 1.25 && ratio <= 1.82;
        if (!portrait && !landscape) continue;

        if (area > bestArea) {
            bestArea = area;
            best = approx;
        }
    }

    if (best.empty()) return false;

    // ── 7. Sub-pixel corner refinement ───────────────────────────────────────
    std::vector<cv::Point2f> corners2f;
    corners2f.reserve(4);
    for (const auto& pt : best)
        corners2f.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));

    cv::cornerSubPix(
        gray, corners2f,
        cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001)
    );

    // ── 8. Sort corners ───────────────────────────────────────────────────────
    // Sort by x+y: index 0 = topLeft (min), index 3 = bottomRight (max)
    std::sort(corners2f.begin(), corners2f.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });

    cv::Point2f topLeft     = corners2f[0];
    cv::Point2f bottomRight = corners2f[3];

    // Of the two middle points, topRight has greater (x - y)
    cv::Point2f topRight, bottomLeft;
    if ((corners2f[1].x - corners2f[1].y) > (corners2f[2].x - corners2f[2].y)) {
        topRight   = corners2f[1];
        bottomLeft = corners2f[2];
    } else {
        topRight   = corners2f[2];
        bottomLeft = corners2f[1];
    }

    float w = static_cast<float>(image.cols);
    float h = static_cast<float>(image.rows);

    out.topLeftX     = topLeft.x     / w;  out.topLeftY     = topLeft.y     / h;
    out.topRightX    = topRight.x    / w;  out.topRightY    = topRight.y    / h;
    out.bottomRightX = bottomRight.x / w;  out.bottomRightY = bottomRight.y / h;
    out.bottomLeftX  = bottomLeft.x  / w;  out.bottomLeftY  = bottomLeft.y  / h;

    return true;
}
