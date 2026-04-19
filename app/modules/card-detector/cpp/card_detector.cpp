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

// Check that a detected quad has a dark outer border vs brighter interior —
// the defining visual signature of an MTG card. Samples a thin band along
// the inside of the quad and compares its mean luma to the interior's.
static bool looksLikeCardBorder(const cv::Mat& gray, const std::vector<cv::Point2f>& pts) {
    cv::Mat maskOuter = cv::Mat::zeros(gray.size(), CV_8UC1);
    std::vector<cv::Point> ipts;
    ipts.reserve(4);
    for (const auto& p : pts) ipts.emplace_back((int)p.x, (int)p.y);
    cv::fillPoly(maskOuter, std::vector<std::vector<cv::Point>>{ipts}, cv::Scalar(255));

    int k = std::max(7, (int)(std::min(gray.cols, gray.rows) * 0.015));
    if (k % 2 == 0) k += 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::Mat maskInterior;
    cv::erode(maskOuter, maskInterior, kernel);
    cv::Mat maskBorder;
    cv::subtract(maskOuter, maskInterior, maskBorder);

    if (cv::countNonZero(maskBorder) == 0 || cv::countNonZero(maskInterior) == 0) return false;
    double borderMean   = cv::mean(gray, maskBorder)[0];
    double interiorMean = cv::mean(gray, maskInterior)[0];
    return borderMean + 10.0 < interiorMean;
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

    // 5. Morphological close — bridges small gaps in the card's outer
    // border much better than simple dilate (dilate-then-erode avoids
    // growing the edge map outward). Kernel scales with image size so
    // the same relative gap is closed at every resolution.
    int ck = std::max(5, (int)(std::min(image.cols, image.rows) * 0.006));
    if (ck % 2 == 0) ck += 1;
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ck, ck));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, closeKernel);

    if (stats) {
        stats->medianLuma = (int)median;
        stats->edgePixels = cv::countNonZero(edges);
    }

    // 6. Find contours — RETR_LIST lets us see contours at every nesting
    // level, not just the outermost. The card's border sometimes is not
    // the topmost contour in noisy scenes.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Real-world photos return hundreds of small noise contours. Keep only
    // the top-N by area — the card is always among the largest.
    if ((int)contours.size() > 40) {
        std::partial_sort(contours.begin(), contours.begin() + 40, contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) > cv::contourArea(b);
            });
        contours.resize(40);
    }

    double imageArea = (double)image.cols * image.rows;
    // Cover-mode display crops a lot of the native frame, so cards often occupy
    // only ~5–8% of the raw 1920×1080 buffer. 3% keeps room for that.
    double minArea   = imageArea * 0.03;
    // Loose cap — the edge-touching check below is the real guard against the
    // video-frame-edge false positive.
    double maxArea   = imageArea * 0.95;
    // Margin (in pixels) for "touches the image edge" rejection.
    int edgeMargin   = std::max(3, (int)(std::min(image.cols, image.rows) * 0.01));

    if (stats) stats->contoursTotal = (int)contours.size();

    // Collect all candidates that pass the geometric filters so we can
    // prefer the top-confidence quad that also passes the card-border
    // darkness check (real MTG cards always have a darker border than
    // their interior).
    struct QuadCandidate {
        std::vector<cv::Point2f> pts;
        float confidence;
    };
    std::vector<QuadCandidate> allCandidates;

    for (const auto& contour : contours) {
        // Convex hull first — collapses edge noise/rounded corners so the
        // following approximation gets near-rectangular polygons.
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);

        // Reject contours whose bounding box touches the image edge —
        // that's the video-frame edge itself, not a real card.
        cv::Rect bbox = cv::boundingRect(hull);
        if (bbox.x < edgeMargin || bbox.y < edgeMargin ||
            bbox.x + bbox.width  > image.cols - edgeMargin ||
            bbox.y + bbox.height > image.rows - edgeMargin) {
            continue;
        }

        double perimeter = cv::arcLength(hull, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(hull, approx, 0.04 * perimeter, true);

        std::vector<cv::Point2f> pts2f;
        if (approx.size() == 4) {
            // Clean 4-vertex quad — use corners directly for best perspective warp.
            for (const auto& p : approx)
                pts2f.push_back(cv::Point2f((float)p.x, (float)p.y));
        } else if (approx.size() >= 5 && approx.size() <= 12) {
            // Card contour sometimes reduces to 5–12 vertices (rounded corners,
            // edge noise, foil effects). Fall back to the min-area rectangle of
            // the hull; require the hull to fill >=80% of that rect so we
            // don't accept random blobs whose bounding rect happens to be big.
            cv::RotatedRect minRect = cv::minAreaRect(hull);
            double rectArea = (double)minRect.size.width * minRect.size.height;
            if (rectArea < 1.0) continue;
            double hullArea = cv::contourArea(hull);
            if (hullArea / rectArea < 0.80) continue;
            cv::Point2f rectPts[4];
            minRect.points(rectPts);
            for (int i = 0; i < 4; i++) pts2f.push_back(rectPts[i]);
        } else {
            continue;
        }
        if (stats) stats->passed4Vertex++;

        double area = cv::contourArea(pts2f);
        if (area < minArea || area > maxArea) continue;
        if (stats) stats->passedMinArea++;

        // Convexity — hull is convex by construction, but approxPolyDP or
        // minAreaRect output should still pass; keep the check as a safety net.
        if (!cv::isContourConvex(pts2f)) continue;
        if (stats) stats->passedConvex++;

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
        // MTG cards are always held portrait in the scanner UI. Landscape
        // rectangles are typically sub-features (rules text boxes, art
        // frames) and cause false positives — reject them.
        bool portrait = ratio >= 0.50f && ratio <= 0.90f;
        if (!portrait) continue;

        // Opposing-edge length ratio check — a real rectangle has parallel,
        // roughly-equal opposing sides. This rejects trapezoid-shaped
        // false positives from threshold segmentation (observed on dark
        // carpets with light glare bands).
        float wMax = std::max(w1, w2), wMin = std::max(std::min(w1, w2), 1.0f);
        float hMax = std::max(h1, h2), hMin = std::max(std::min(h1, h2), 1.0f);
        if (wMax / wMin > 1.3f || hMax / hMin > 1.3f) continue;
        if (stats) stats->passedAR++;

        // Confidence: 60% area + 20% angle + 20% AR. Portrait target AR = 0.715
        // (standard MTG card). Area saturates at 25% image coverage — typical
        // hand-held card captures sit at 15–25% so this gives real cards a
        // near-full area score without letting full-frame quads dominate.
        float areaScore  = (float)std::min(1.0, area / (imageArea * 0.25));
        float angleScore = std::max(0.0f, 1.0f - (totalAngleDev / 4.0f) / 20.0f);
        float arScore    = std::max(0.0f, 1.0f - std::abs(ratio - 0.715f) / 0.165f);
        float confidence = 0.60f * areaScore + 0.20f * angleScore + 0.20f * arScore;

        allCandidates.push_back({sorted, confidence});
    }

    if (allCandidates.empty()) return false;

    // Prefer the highest-confidence candidate that also passes the
    // card-border darkness check. Fall back to raw confidence if no
    // candidate passes (better to return something than nothing).
    std::sort(allCandidates.begin(), allCandidates.end(),
        [](const QuadCandidate& a, const QuadCandidate& b) {
            return a.confidence > b.confidence;
        });

    std::vector<cv::Point2f> bestPts;
    float bestConf = -1.0f;
    for (const auto& c : allCandidates) {
        if (looksLikeCardBorder(gray, c.pts)) {
            bestPts = c.pts;
            bestConf = c.confidence;
            break;
        }
    }
    if (bestPts.empty()) {
        bestPts = allCandidates[0].pts;
        bestConf = allCandidates[0].confidence;
    }

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
