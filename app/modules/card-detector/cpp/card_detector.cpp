#include "card_detector.h"
#include <algorithm>
#include <cmath>
#include <vector>

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

static float angleDeg(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
    cv::Point2f v1 = a - b, v2 = c - b;
    float len1 = cv::norm(v1), len2 = cv::norm(v2);
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    float cosA = v1.dot(v2) / (len1 * len2);
    cosA = std::max(-1.0f, std::min(1.0f, cosA));
    return std::acos(cosA) * 180.0f / (float)M_PI;
}

// Sort 4 float points to [topLeft, topRight, bottomRight, bottomLeft]
static std::vector<cv::Point2f> sortCorners(std::vector<cv::Point2f> pts) {
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });
    cv::Point2f tl = pts[0], br = pts[3], tr, bl;
    if ((pts[1].x - pts[1].y) > (pts[2].x - pts[2].y)) {
        tr = pts[1]; bl = pts[2];
    } else {
        tr = pts[2]; bl = pts[1];
    }
    return {tl, tr, br, bl};
}

bool detectCardCorners(const cv::Mat& image, CardCorners& out,
                       std::string* rectifiedPath, DetectionStats* stats) {
    if (stats) *stats = {};
    if (image.empty()) return false;

    // 1. Grayscale
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 2. CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat equalized;
    clahe->apply(gray, equalized);

    // 3. Bilateral filter — edge-preserving smoothing
    // Use lighter params on frame processor path (no rectifiedPath) to stay under 30fps budget
    cv::Mat filtered;
    if (rectifiedPath) {
        cv::bilateralFilter(equalized, filtered, 9, 75, 75);   // photo path: quality
    } else {
        cv::bilateralFilter(equalized, filtered, 5, 50, 50);   // frame path: speed
    }

    // 4. Adaptive Canny
    double median = imageMedian(filtered);
    double lo = std::max(0.0, 0.67 * median);
    double hi = std::min(255.0, 1.33 * median);
    cv::Mat edges;
    cv::Canny(filtered, edges, lo, hi);

    // 5. Dilation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 1);

    if (stats) {
        stats->medianLuma = (int)median;
        stats->edgePixels = cv::countNonZero(edges);
    }

    // 6. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double imageArea = (double)image.cols * image.rows;
    // Cover-mode display crops a lot of the native frame, so cards often occupy
    // only ~5–8% of the raw 1920×1080 buffer. 3% keeps room for that.
    double minArea   = imageArea * 0.03;

    if (stats) stats->contoursTotal = (int)contours.size();

    std::vector<cv::Point2f> bestPts;
    float bestConf = -1.0f;

    for (const auto& contour : contours) {
        double perimeter = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.025 * perimeter, true);
        if (approx.size() != 4) continue;
        if (stats) stats->passed4Vertex++;

        double area = cv::contourArea(approx);
        if (area < minArea) continue;
        if (stats) stats->passedMinArea++;

        // Convexity check
        if (!cv::isContourConvex(approx)) continue;
        if (stats) stats->passedConvex++;

        std::vector<cv::Point2f> pts2f;
        pts2f.reserve(4);
        for (const auto& p : approx)
            pts2f.push_back(cv::Point2f((float)p.x, (float)p.y));

        auto sorted = sortCorners(pts2f);

        // Interior angle validation (60–120°, widened for perspective skew)
        bool anglesOk = true;
        float totalAngleDev = 0.0f;
        for (int i = 0; i < 4; i++) {
            float angle = angleDeg(sorted[(i + 3) % 4], sorted[i], sorted[(i + 1) % 4]);
            if (angle < 60.0f || angle > 120.0f) { anglesOk = false; break; }
            totalAngleDev += std::abs(angle - 90.0f);
        }
        if (!anglesOk) continue;
        if (stats) stats->passedAngles++;

        // Quad-edge aspect ratio (actual edge lengths, not bounding box)
        float w1 = cv::norm(sorted[1] - sorted[0]);
        float w2 = cv::norm(sorted[2] - sorted[3]);
        float h1 = cv::norm(sorted[3] - sorted[0]);
        float h2 = cv::norm(sorted[2] - sorted[1]);
        float avgW = (w1 + w2) / 2.0f;
        float avgH = (h1 + h2) / 2.0f;
        if (avgH < 1.0f) continue;
        float ratio = avgW / avgH;
        bool portrait  = ratio >= 0.50f && ratio <= 0.90f;
        bool landscape = ratio >= 1.10f && ratio <= 2.00f;
        if (!portrait && !landscape) continue;
        if (stats) stats->passedAR++;

        // Confidence: 40% area + 30% angle + 30% AR
        float areaScore  = (float)std::min(1.0, area / (imageArea * 0.5));
        float angleScore = std::max(0.0f, 1.0f - (totalAngleDev / 4.0f) / 20.0f);
        float targetAR   = portrait ? 0.715f : 1.535f;
        float arRange    = portrait ? 0.165f : 0.285f;
        float arScore    = std::max(0.0f, 1.0f - std::abs(ratio - targetAR) / arRange);
        float confidence = 0.40f * areaScore + 0.30f * angleScore + 0.30f * arScore;

        if (confidence > bestConf) {
            bestConf = confidence;
            bestPts  = sorted;
        }
    }

    if (bestPts.empty()) return false;

    // 7. Sub-pixel refinement
    try {
        cv::cornerSubPix(
            gray, bestPts,
            cv::Size(5, 5), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001)
        );
    } catch (const cv::Exception&) {
        // corner too close to border — use integer-precision corners
    }

    // 8. Perspective warp to 400×560
    if (rectifiedPath && !rectifiedPath->empty()) {
        std::vector<cv::Point2f> src = bestPts;
        std::vector<cv::Point2f> dst = {{0,0}, {400,0}, {400,560}, {0,560}};
        cv::Mat M = cv::getPerspectiveTransform(src, dst);
        if (std::abs(cv::determinant(M)) > 1e-6) {
            cv::Mat rectified;
            cv::warpPerspective(image, rectified, M, cv::Size(400, 560));
            if (!cv::imwrite(*rectifiedPath, rectified)) {
                rectifiedPath->clear();  // signal to caller that file was not written
            }
        } else {
            rectifiedPath->clear();
        }
    }

    // 9. Normalize to [0,1]
    float w = (float)image.cols, h = (float)image.rows;
    out.topLeftX     = bestPts[0].x / w;  out.topLeftY     = bestPts[0].y / h;
    out.topRightX    = bestPts[1].x / w;  out.topRightY    = bestPts[1].y / h;
    out.bottomRightX = bestPts[2].x / w;  out.bottomRightY = bestPts[2].y / h;
    out.bottomLeftX  = bestPts[3].x / w;  out.bottomLeftY  = bestPts[3].y / h;
    out.confidence   = bestConf;
    return true;
}
