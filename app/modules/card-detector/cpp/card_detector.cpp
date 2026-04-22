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
//may need to update. could struggle with some special printings
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

// Scale corners about their centroid. Positive pct grows the quad,
// negative shrinks it. Used to compensate for per-source detection bias
// before perspective warp — Scryfall-trained encoder expects a
// `normal`-image-sized border band.
static std::vector<cv::Point2f> expandQuad(const std::vector<cv::Point2f>& pts, float pct) {
    cv::Point2f ctr(0, 0);
    for (const auto& p : pts) ctr += p;
    ctr *= (1.0f / (float)pts.size());
    std::vector<cv::Point2f> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.push_back(ctr + (p - ctr) * (1.0f + pct));
    return out;
}

// ── line-pair fallback ────────────────────────────────────────────────────
//
// Pattern: an MTG card has multiple parallel horizontal rules (title
// banner, type line, text-box rules) and two vertical edges. After
// clustering sub-pixel-duplicate Hough segments into physical edges,
// the outermost pair on each axis bounds the card directly — no
// aspect-ratio extrapolation required.

struct HoughSeg {
    int x1, y1, x2, y2;
    float length;
    float midX, midY;
};

struct LineCluster {
    float coord;       // mean perpendicular coord (y for H, x for V)
    float totalLen;    // sum of segment lengths
    cv::Point2f repP1, repP2;  // longest segment's endpoints (for debug)
};

// Merge segments by perpendicular midpoint coordinate within `tol`.
// getCoord selects the axis (mid_y for H, mid_x for V).
static std::vector<LineCluster> clusterByPerpendicular(
    std::vector<HoughSeg> segs,
    bool byY,
    float tol
) {
    if (segs.empty()) return {};
    // Use stable_sort to match Python's sorted() — ties on midY/midX
    // (possible when Hough returns near-duplicate segments) are resolved
    // by original order rather than introsort's undefined tie-break.
    std::stable_sort(segs.begin(), segs.end(), [byY](const HoughSeg& a, const HoughSeg& b) {
        return (byY ? a.midY : a.midX) < (byY ? b.midY : b.midX);
    });
    std::vector<std::vector<HoughSeg>> groups;
    groups.push_back({segs[0]});
    for (size_t i = 1; i < segs.size(); ++i) {
        const float c = byY ? segs[i].midY : segs[i].midX;
        float mean = 0.0f;
        for (const auto& s : groups.back()) mean += byY ? s.midY : s.midX;
        mean /= (float)groups.back().size();
        if (c - mean < tol) groups.back().push_back(segs[i]);
        else groups.push_back({segs[i]});
    }
    std::vector<LineCluster> out;
    out.reserve(groups.size());
    for (const auto& g : groups) {
        LineCluster c{};
        float sumCoord = 0.0f;
        const HoughSeg* longest = &g[0];
        for (const auto& s : g) {
            sumCoord += byY ? s.midY : s.midX;
            c.totalLen += s.length;
            if (s.length > longest->length) longest = &s;
        }
        c.coord = sumCoord / (float)g.size();
        c.repP1 = cv::Point2f((float)longest->x1, (float)longest->y1);
        c.repP2 = cv::Point2f((float)longest->x2, (float)longest->y2);
        out.push_back(c);
    }
    return out;
}

// Try to find the card from clustered Hough line intersections when
// the primary contour path fails. Returns non-empty pts + confidence
// on success.
static bool detectViaLinePair(
    const cv::Mat& gray, const cv::Mat& edges,
    std::vector<cv::Point2f>& outPts, float& outConf
) {
    const int W = gray.cols, H = gray.rows;
    const int minSegLen   = (int)(std::min(W, H) * 0.25);
    const int maxLineGap  = (int)(std::min(W, H) * 0.015);
    const int margin      = std::max(2, (int)(std::min(W, H) * 0.02));
    const int houghThresh = 40;

    std::vector<cv::Vec4i> linesRaw;
    cv::HoughLinesP(edges, linesRaw, 1, CV_PI / 180, houghThresh,
                    (double)minSegLen, (double)maxLineGap);
    if (linesRaw.empty()) return false;

    std::vector<HoughSeg> hSegs, vSegs;
    for (const auto& l : linesRaw) {
        int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        float len = (float)std::hypot(x2 - x1, y2 - y1);
        if (len < (float)minSegLen) continue;
        float deg = (float)(std::atan2((double)std::abs(y2 - y1),
                                       (double)std::abs(x2 - x1)) * 180.0 / CV_PI);
        float mx = (x1 + x2) / 2.0f, my = (y1 + y2) / 2.0f;
        if (mx < margin || mx > W - margin || my < margin || my > H - margin) continue;
        HoughSeg s{x1, y1, x2, y2, len, mx, my};
        if (deg <= 8.0f) hSegs.push_back(s);
        else if (deg >= 82.0f) vSegs.push_back(s);
    }
    if (hSegs.empty() || vSegs.empty()) return false;

    const float clusterTol = std::max(5.0f, std::min(W, H) * 0.02f);
    auto hClusters = clusterByPerpendicular(hSegs, true, clusterTol);
    auto vClusters = clusterByPerpendicular(vSegs, false, clusterTol);
    if (hClusters.size() < 2 || vClusters.size() < 2) return false;

    // Enumerate every (h_top, h_bot) × (v_left, v_right) rectangle;
    // filter by card AR [0.60, 0.85] + minimum size; accept if
    // looksLikeCardBorder passes; score by area · (0.4 + 0.3·ar_score + 0.3·cov).
    struct BestR { std::vector<cv::Point2f> pts; float score, ratio, cw, ch, coverage; };
    BestR best{{}, -1.0f, 0, 0, 0, 0};

    const float imageArea = (float)(W * H);
    for (size_t i = 0; i < hClusters.size(); ++i) {
        for (size_t j = i + 1; j < hClusters.size(); ++j) {
            const float ch = hClusters[j].coord - hClusters[i].coord;
            if (ch < 0.35f * H) continue;
            for (size_t k = 0; k < vClusters.size(); ++k) {
                for (size_t m = k + 1; m < vClusters.size(); ++m) {
                    const float cw = vClusters[m].coord - vClusters[k].coord;
                    if (cw < 0.25f * W) continue;
                    const float ratio = cw / ch;
                    if (ratio < 0.60f || ratio > 0.85f) continue;
                    std::vector<cv::Point2f> corners = {
                        {vClusters[k].coord, hClusters[i].coord},
                        {vClusters[m].coord, hClusters[i].coord},
                        {vClusters[m].coord, hClusters[j].coord},
                        {vClusters[k].coord, hClusters[j].coord},
                    };
                    // Clip for border check
                    std::vector<cv::Point2f> clipped = corners;
                    for (auto& p : clipped) {
                        p.x = std::max(0.0f, std::min((float)(W - 1), p.x));
                        p.y = std::max(0.0f, std::min((float)(H - 1), p.y));
                    }
                    if (!looksLikeCardBorder(gray, clipped)) continue;
                    const float areaFrac = (cw * ch) / imageArea;
                    const float arScore  = std::max(0.0f, 1.0f - std::abs(ratio - 0.716f) / 0.125f);
                    const float covH = std::min(1.0f,
                        (hClusters[i].totalLen + hClusters[j].totalLen) / (2.0f * cw));
                    const float covV = std::min(1.0f,
                        (vClusters[k].totalLen + vClusters[m].totalLen) / (2.0f * ch));
                    const float coverage = (covH + covV) * 0.5f;
                    const float score = areaFrac * (0.4f + 0.3f * arScore + 0.3f * coverage);
                    if (score > best.score) {
                        best = {corners, score, ratio, cw, ch, coverage};
                    }
                }
            }
        }
    }

    if (best.score < 0.0f) return false;

    const float arScore = std::max(0.0f, 1.0f - std::abs(best.ratio - 0.716f) / 0.125f);
    outConf = std::min(0.90f, 0.55f + 0.20f * best.coverage + 0.15f * arScore);
    outPts = best.pts;
    return true;
}

// ── Otsu fallback ─────────────────────────────────────────────────────────
//
// Adapted from https://stackoverflow.com/a/55874079 (nathancy, CC BY-SA 4.0).
// Otsu splits foreground/background on luma distribution — orthogonal
// signal to Canny's gradient detection. Works on low-contrast outer
// edges that Canny misses (card on textured carpet, card on
// screenshot). Tries both polarities; best scoring contour wins.
static bool detectViaOtsu(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& outPts, float& outConf
) {
    const int W = gray.cols, H = gray.rows;
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);

    const float imageArea = (float)(W * H);
    const float minArea   = imageArea * 0.08f;
    const float maxArea   = imageArea * 0.95f;
    // Match Python _detect_via_otsu exactly — no oddification. OpenCV's
    // MORPH_CLOSE accepts even kernels; forcing odd diverged results on
    // images where 0.006·min(W,H) rounds to an even int.
    const int   ck = std::max(5, (int)(std::min(W, H) * 0.006));
    cv::Mat closeK = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ck, ck));

    struct BestO { std::vector<cv::Point2f> pts; float score, ratio, areaFrac; };
    BestO best{{}, -1.0f, 0, 0};

    const int flags[2] = { cv::THRESH_BINARY, cv::THRESH_BINARY_INV };
    for (int fi = 0; fi < 2; ++fi) {
        cv::Mat thresh;
        cv::threshold(blur, thresh, 0, 255, flags[fi] | cv::THRESH_OTSU);
        cv::Mat cleaned;
        cv::morphologyEx(thresh, cleaned, cv::MORPH_CLOSE, closeK);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& c : contours) {
            const double area = cv::contourArea(c);
            if (area < minArea || area > maxArea) continue;
            std::vector<cv::Point> hull;
            cv::convexHull(c, hull);
            cv::Rect bbox = cv::boundingRect(hull);
            // Reject only the whole-image spurious contour (bbox spans
            // ≥95% of both axes). Rejecting any edge-touching quad
            // throws out cards that fill the frame — common for phone
            // photos, e.g. IMG_0472 (Retrieval Agent on carpet) covers
            // ~85% of the frame and its bbox abuts all four edges.
            if (bbox.width >= (int)(0.95 * W) && bbox.height >= (int)(0.95 * H)) continue;
            cv::RotatedRect rr = cv::minAreaRect(hull);
            const float rw = rr.size.width, rh = rr.size.height;
            if (rw < 1.0f || rh < 1.0f) continue;
            const double hullArea = cv::contourArea(hull);
            if (hullArea / (rw * rh) < 0.75) continue;
            const float ratio = std::min(rw, rh) / std::max(rw, rh);
            // AR upper bound 0.88 for Otsu. Card-on-low-contrast-
            // background bleeds into the foreground blob and widens
            // minAreaRect perpendicular to the card (observed 0.851
            // on IMG_0472 carpet fixture). At 0.88 the text-box false
            // positive still loses on score since the whole-card area
            // advantage dominates even when ar_score is 0.
            if (ratio < 0.60f || ratio > 0.88f) continue;

            // Snap minAreaRect aspect to card AR (0.716) by shrinking
            // the wider axis about the rect centre. Otsu foreground
            // blobs bleed into low-contrast backgrounds (card on
            // carpet / matching screenshot), widening the bbox beyond
            // the true card outline; warping the widened rect to
            // fixed 400×560 squashes the card horizontally and the
            // encoder can't match it. Applied unconditionally — no
            // reliable threshold separates bleed cases from tight
            // segmentations using AR alone.
            cv::RotatedRect corrected = rr;
            if (rw < rh) {
                corrected.size.width  = (float)(rh * (63.0 / 88.0));
            } else {
                corrected.size.height = (float)(rw * (63.0 / 88.0));
            }
            cv::Point2f boxPts[4];
            corrected.points(boxPts);
            std::vector<cv::Point2f> raw(boxPts, boxPts + 4);
            std::vector<cv::Point2f> sorted = sortCorners(raw);

            // No dark-border gate — the Otsu segmentation is already the
            // foreground/background evidence. Darker-on-bright and
            // brighter-on-dark card orientations both pass.
            const float arScore  = std::max(0.0f, 1.0f - std::abs(ratio - 0.716f) / 0.125f);
            const float areaFrac = (float)area / imageArea;
            const float score    = areaFrac * (0.5f + 0.5f * arScore);
            if (score > best.score) {
                best = {sorted, score, ratio, areaFrac};
            }
        }
    }

    if (best.score < 0.0f) return false;
    const float arScore = std::max(0.0f, 1.0f - std::abs(best.ratio - 0.716f) / 0.125f);
    outConf = std::min(0.90f, 0.55f + 0.20f * best.areaFrac + 0.15f * arScore);
    outPts = best.pts;
    return true;
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

    std::vector<cv::Point2f> bestPts;
    float bestConf = -1.0f;
    int   bestSource = CARD_SOURCE_PRIMARY;

    if (!allCandidates.empty()) {
        // Prefer the highest-confidence candidate that also passes the
        // card-border darkness check. Fall back to raw confidence if no
        // candidate passes (better to return something than nothing).
        std::sort(allCandidates.begin(), allCandidates.end(),
            [](const QuadCandidate& a, const QuadCandidate& b) {
                return a.confidence > b.confidence;
            });

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

        // 7. Sub-pixel refinement (primary path only).
        try {
            cv::cornerSubPix(
                gray, bestPts,
                cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001)
            );
        } catch (const cv::Exception&) {
            // corner too close to border — use integer-precision corners
        }
    } else {
        // Primary path found nothing. Try the Hough line-pair fallback,
        // then the Otsu segmentation fallback. First success wins.
        if (detectViaLinePair(gray, edges, bestPts, bestConf)) {
            bestSource = CARD_SOURCE_LINEINTERP;
        } else if (detectViaOtsu(gray, bestPts, bestConf)) {
            bestSource = CARD_SOURCE_OTSU;
        } else {
            return false;
        }
    }

    // 8. Perspective warp to 400×560
    if (rectifiedPath && !rectifiedPath->empty()) {
        // Per-source expansion about quad centroid, chosen to match the
        // Scryfall `normal` training-image framing:
        //   primary    ( 0%)  contour fits the card exactly
        //   lineinterp (+5%)  rectangle is inset from outer edge
        //                     (looksLikeCardBorder snaps to interior)
        //   otsu       (−5%)  minAreaRect on Otsu fg usually includes
        //                     a narrow background halo
        float expandPct = 0.0f;
        if      (bestSource == CARD_SOURCE_LINEINTERP) expandPct = +0.05f;
        else if (bestSource == CARD_SOURCE_OTSU)       expandPct = -0.05f;
        std::vector<cv::Point2f> src = (expandPct == 0.0f)
            ? bestPts
            : expandQuad(bestPts, expandPct);
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

    // 9. Normalize corner coords to [0,1] for the JS layer.
    float w = (float)image.cols, h = (float)image.rows;
    out.topLeftX     = bestPts[0].x / w;  out.topLeftY     = bestPts[0].y / h;
    out.topRightX    = bestPts[1].x / w;  out.topRightY    = bestPts[1].y / h;
    out.bottomRightX = bestPts[2].x / w;  out.bottomRightY = bestPts[2].y / h;
    out.bottomLeftX  = bestPts[3].x / w;  out.bottomLeftY  = bestPts[3].y / h;
    out.confidence   = bestConf;
    out.source       = bestSource;
    return true;
}
