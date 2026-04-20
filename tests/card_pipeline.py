"""
Python mirror of the on-device detection + OCR pipeline.

Keeps the same stages, filter thresholds, and crop regions as the C++/TS
versions so improvements made here port directly back to the device.

Pipeline:
  1. grayscale → CLAHE → bilateral filter
  2. adaptive Canny (median-based) → dilate
  3. findContours → convex hull
  4. approxPolyDP (0.04 * perimeter) OR minAreaRect fallback (5-12 vertex hull)
  5. reject edge-touching contours (video frame edge)
  6. portrait AR 0.50-0.90 → interior angles 60-120° → area 3-95%
  7. highest confidence wins
  8. perspective warp → 400x560
  9. OCR fixed pixel crop for set/collector, fallback name crop
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract


# ── constants (mirror card_detector.cpp / ocr.ts) ────────────────────────────

RECT_W, RECT_H = 400, 560
# Modern frame lower-left info block: 2 lines (COLLECTOR/TOTAL RARITY,
# then SET LANG DESIGNER). Covers the vast majority of MTG prints.
CROP_BOTTOM_LEFT  = (5, 515, 220, 45)   # x, y, w, h
# Thin full-width strip catches retro-frame cards (Swiftfoot Boots BRO retro)
# where the collector number is at the far RIGHT of the copyright line.
CROP_RETRO_BOTTOM = (5, 548, 395, 15)
# Card name along the top.
CROP_NAME = (16, 22, 370, 40)

MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.95
EDGE_MARGIN_FRAC = 0.01

# Set codes that real MTG sets have never used and therefore should be
# discarded from OCR noise. Expanded list copies the app's SKIP_TOKENS.
_SKIP_TOKENS = {
    'EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'JA', 'KO', 'RU', 'ZH', 'PH', 'CS',
    'R', 'U', 'C', 'M', 'S', 'T', 'L',
    'THE', 'AND', 'FOR', 'YOU', 'MAY', 'TAP', 'PUT', 'TOP', 'NEW',
    'COPY', 'CAST', 'EACH', 'FROM', 'YOUR', 'BEEN', 'THAT', 'CARD',
    'WITH', 'LESS', 'MANA', 'AURA', 'NON',
    'MEE', 'LLC', 'INC', 'LTD', 'ALL', 'TM',
}


@dataclass
class ScanResult:
    corners:          Optional[np.ndarray]   # (4,2) tl,tr,br,bl
    confidence:       float
    rectified:        Optional[np.ndarray]   # 400x560 BGR
    bl_text:          str
    name_text:        str
    set_code:         Optional[str]
    collector_number: Optional[str]
    card_name:        Optional[str] = None   # populated only if query_scryfall=True


# ── detection (ported from C++ card_detector.cpp) ────────────────────────────

def _image_median(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size
    cum = 0
    for i, count in enumerate(hist):
        cum += int(count)
        if cum >= total // 2:
            return float(i)
    return 128.0


def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 float points to tl, tr, br, bl."""
    s = pts.sum(axis=1)
    idx = np.argsort(s)
    tl, br = pts[idx[0]], pts[idx[3]]
    m1, m2 = pts[idx[1]], pts[idx[2]]
    # of the two middle points, tr has larger (x - y)
    if (m1[0] - m1[1]) > (m2[0] - m2[1]):
        tr, bl = m1, m2
    else:
        tr, bl = m2, m1
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _angle_deg(a, b, c) -> float:
    v1, v2 = a - b, c - b
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 < 1e-6 or l2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def _build_edge_map_canny(image: np.ndarray, close_fraction: float = 0.006) -> np.ndarray:
    """Canny-based edge map with adaptive thresholds and morphological
    closing to bridge small gaps in the card border."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    filtered = cv2.bilateralFilter(eq, 9, 75, 75)
    median = _image_median(filtered)
    edges = cv2.Canny(filtered, max(0.0, 0.67 * median), min(255.0, 1.33 * median))
    k = max(5, int(min(image.shape[:2]) * close_fraction))
    if k % 2 == 0: k += 1
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def _build_edge_map_threshold(image: np.ndarray) -> np.ndarray:
    """Threshold-based fallback — isolates bright regions (card interior /
    title bar) and closes them into a solid blob. Works when the outer
    card border is obscured by fingers or low contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Otsu to separate bright card body from darker surroundings
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(11, int(min(image.shape[:2]) * 0.02))
    if k % 2 == 0: k += 1
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # Contours of filled regions look like solid card blobs — convert to
    # "edges" of those blobs for the shared pipeline.
    return closed


def _looks_like_card_border(gray: np.ndarray, pts: np.ndarray) -> bool:
    """MTG cards have a dark outer border. Sample a thin band along the
    inside of the quad; mean gray must be meaningfully darker than the
    quad interior's mean gray."""
    mask_outer = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask_outer, [pts.astype(np.int32)], 255)
    # shrink mask to get interior region
    k = max(7, int(min(gray.shape) * 0.015))
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask_interior = cv2.erode(mask_outer, kernel, iterations=1)
    mask_border = cv2.subtract(mask_outer, mask_interior)

    if cv2.countNonZero(mask_border) == 0 or cv2.countNonZero(mask_interior) == 0:
        return False
    border_mean = cv2.mean(gray, mask=mask_border)[0]
    interior_mean = cv2.mean(gray, mask=mask_interior)[0]
    # Require the border strip to be at least ~10 gray levels darker than
    # the interior. Cards always satisfy this; background false positives
    # usually don't.
    return border_mean + 10 < interior_mean


def detect_card_corners(image: np.ndarray, stats: Optional[dict] = None) -> Tuple[Optional[np.ndarray], float]:
    if image is None or image.size == 0:
        return None, 0.0
    if stats is None:
        stats = {}
    stats.update({
        'contours': 0, 'rejected_edge_touch': 0,
        'rejected_vertex_count': 0, 'rejected_rectangularity': 0,
        'rejected_area': 0, 'rejected_convex': 0,
        'rejected_angle': 0, 'rejected_ar': 0, 'passed': 0,
    })

    # Try multiple preprocessing variants — small kernel for clean cards,
    # medium for partial occlusion. Threshold path handles low-contrast.
    candidates = []
    for em in (_build_edge_map_canny(image, close_fraction=0.004),
               _build_edge_map_canny(image, close_fraction=0.010),
               _build_edge_map_threshold(image)):
        b = _find_best_quad(image, em, stats)
        if b is not None:
            candidates.append(b)
    if not candidates:
        return None, 0.0
    # Prefer candidates that pass the card-border darkness check; fall back
    # to raw confidence if none do.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    card_like = [c for c in candidates if _looks_like_card_border(gray_img, c[0])]
    pool = card_like if card_like else candidates
    pts, conf = max(pool, key=lambda c: c[1])

    # sub-pixel refinement on the pre-CLAHE gray for stability
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        refined = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (5, 5), (-1, -1), criteria)
        pts = refined.reshape(4, 2)
    except cv2.error:
        pass
    return pts, conf


def _find_best_quad(image: np.ndarray, edges: np.ndarray, stats: dict) -> Optional[Tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    h, w = image.shape[:2]
    image_area = float(w * h)
    min_area = image_area * MIN_AREA_FRAC
    max_area = image_area * MAX_AREA_FRAC
    edge_margin = max(3, int(min(w, h) * EDGE_MARGIN_FRAC))

    best_pts: Optional[np.ndarray] = None
    best_conf = -1.0
    stats['contours'] = max(stats.get('contours', 0), len(contours))

    for contour in contours:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)

        if len(approx) == 4:
            pts2f = approx.reshape(4, 2).astype(np.float32)
        elif 5 <= len(approx) <= 12:
            rect = cv2.minAreaRect(hull)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area < 1.0:
                stats['rejected_rectangularity'] += 1
                continue
            hull_area = cv2.contourArea(hull)
            if hull_area / rect_area < 0.80:
                stats['rejected_rectangularity'] += 1
                continue
            pts2f = cv2.boxPoints(rect).astype(np.float32)
        else:
            stats['rejected_vertex_count'] += 1
            continue

        area = cv2.contourArea(pts2f)
        if area < min_area or area > max_area:
            stats['rejected_area'] += 1
            continue

        if not cv2.isContourConvex(pts2f.astype(np.int32)):
            stats['rejected_convex'] += 1
            continue

        sorted_pts = _sort_corners(pts2f)

        # angle check 60-120°
        angles_ok = True
        total_dev = 0.0
        for i in range(4):
            a = sorted_pts[(i + 3) % 4]
            b = sorted_pts[i]
            c = sorted_pts[(i + 1) % 4]
            ang = _angle_deg(a, b, c)
            if ang < 60.0 or ang > 120.0:
                angles_ok = False
                break
            total_dev += abs(ang - 90.0)
        if not angles_ok:
            stats['rejected_angle'] += 1
            continue

        # AR check — portrait only
        w1 = np.linalg.norm(sorted_pts[1] - sorted_pts[0])
        w2 = np.linalg.norm(sorted_pts[2] - sorted_pts[3])
        h1 = np.linalg.norm(sorted_pts[3] - sorted_pts[0])
        h2 = np.linalg.norm(sorted_pts[2] - sorted_pts[1])
        avg_w = (w1 + w2) / 2
        avg_h = (h1 + h2) / 2
        if avg_h < 1.0:
            stats['rejected_ar'] += 1
            continue
        ratio = avg_w / avg_h
        if not (0.50 <= ratio <= 0.90):
            stats['rejected_ar'] += 1
            continue

        # Opposing-edge length check — a real rectangle has parallel,
        # roughly-equal opposing sides. Rejects trapezoid-shaped false
        # positives from noisy threshold segmentation.
        if max(w1, w2) / max(min(w1, w2), 1.0) > 1.3 or \
           max(h1, h2) / max(min(h1, h2), 1.0) > 1.3:
            stats['rejected_ar'] += 1
            continue

        stats['passed'] += 1
        # confidence
        area_score = min(1.0, area / (image_area * 0.25))
        angle_score = max(0.0, 1.0 - (total_dev / 4.0) / 20.0)
        ar_score = max(0.0, 1.0 - abs(ratio - 0.715) / 0.165)
        confidence = 0.60 * area_score + 0.20 * angle_score + 0.20 * ar_score

        if confidence > best_conf:
            best_conf = confidence
            best_pts = sorted_pts

    if best_pts is None:
        return None
    return (best_pts, best_conf)


def rectify(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (RECT_W, RECT_H))


# ── OCR + parsing ────────────────────────────────────────────────────────────

def _crop(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def _preprocess_for_ocr(crop: np.ndarray, invert: bool = False) -> np.ndarray:
    """Upscale + threshold — tesseract performs much better on high-contrast
    images that are at least ~400px on the long edge."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = 6
    up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (3, 3), 0)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
    return th


def run_ocr(crop: np.ndarray) -> str:
    """Single-config OCR — deterministic and predictable for the caller."""
    pre = _preprocess_for_ocr(crop)
    return pytesseract.image_to_string(
        pre, lang='eng', config='--oem 1 --psm 6').strip()


def _ocr_variants(crop: np.ndarray):
    """Generator of (text, variant_id) across OCR configurations.
    Callers iterate and pick the first that parses successfully."""
    for invert in (False, True):
        pre = _preprocess_for_ocr(crop, invert=invert)
        for psm in (6, 7, 11):
            txt = pytesseract.image_to_string(
                pre, lang='eng', config=f'--oem 1 --psm {psm}').strip()
            yield txt, f'inv={invert}/psm={psm}'


def parse_set_and_number(text: str) -> Optional[Tuple[str, str]]:
    """Pick the best (set, collector) pair from OCR text.

    For each OCR token we extract up to three 3-char candidates: the
    whole token if length 3, its 3-char prefix, and its 3-char suffix.
    Each candidate is matched against the real Scryfall set list; we
    prefer candidates whose position is AFTER the collector number in
    the token stream, matching the physical card layout
    'COLLECTOR/TOTAL RARITY SET LANG'.
    """
    tokens = [t for t in re.split(r'[^A-Za-z0-9/]+', text.upper()) if t]

    # Collect all digit-run candidates, skip obvious copyright years.
    raw_collectors = []
    for i, tok in enumerate(tokens):
        val = None
        if re.fullmatch(r'\d+', tok):
            val = tok
        elif re.fullmatch(r'\d+/\d+', tok):
            val = tok.split('/')[0]
        if val is None:
            continue
        # Reject 4-digit copyright-year patterns (2010–2029).
        if re.fullmatch(r'20[0-2]\d', val):
            continue
        raw_collectors.append((val, i))
    if not raw_collectors:
        return None
    collector, collector_pos = raw_collectors[0]

    try:
        from _set_codes import known_set_codes
        known = known_set_codes()
    except Exception:
        known = None

    def looks_set(s: str) -> bool:
        if s in _SKIP_TOKENS:
            return False
        if not re.fullmatch(r'[A-Z][A-Z0-9]{2}', s):
            return False
        return known is None or s.lower() in known

    # On real cards the set code is always immediately followed by the
    # language code ("BRO EN", "SOC • EN"). Use that as a ranking signal —
    # a candidate with a language code right after it is much more likely
    # to be the real set than one without.
    LANG_CODES = {'EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'JA', 'KO', 'RU', 'ZH', 'PH'}

    def lang_score(tok_idx: int, tok: str, set_str: str) -> int:
        # 1) set_str is a prefix of tok — check rest of tok for lang code
        if tok.startswith(set_str) and len(tok) > 3:
            if tok[3:5] in LANG_CODES:
                return 10
            if any(lc in tok[3:8] for lc in LANG_CODES):
                return 5
        # 2) Next token is a language code
        if tok_idx + 1 < len(tokens) and tokens[tok_idx + 1][:2] in LANG_CODES:
            return 10
        return 0

    candidates: list = []   # (set_code, token_pos, lang_score)
    for pos, tok in enumerate(tokens):
        slices = set()
        if len(tok) == 3:
            slices.add(tok)
        if len(tok) >= 3:
            slices.add(tok[:3])
            slices.add(tok[-3:])
        for s in slices:
            if looks_set(s):
                candidates.append((s, pos, lang_score(pos, tok, s)))

    # Fallback for OCR that inserts stray spaces (e.g. 'S OC' for 'SOC').
    # Scan the letters-only text for a known set code that's followed by
    # a language code — the 'SET•LANG' pattern. That specificity keeps
    # the false-positive rate low.
    if not candidates and known is not None:
        letters = re.sub(r'[^A-Z]', '', text.upper())
        for i in range(len(letters) - 4):
            s = letters[i:i + 3]
            if s in _SKIP_TOKENS or s.lower() not in known:
                continue
            following = letters[i + 3:i + 8]
            if any(lc in following for lc in LANG_CODES):
                candidates.append((s, 10_000 + i, 10))
                break

    if not candidates:
        return None

    # Rank: highest lang_score first, then earliest candidate AFTER the
    # collector number, then earliest overall.
    def sort_key(c):
        set_s, pos, lscore = c
        after_collector = pos > collector_pos
        return (-lscore, 0 if after_collector else 1, pos)

    candidates.sort(key=sort_key)
    chosen_set = candidates[0][0]

    # If the collector is 5+ digits, OCR almost certainly swallowed a
    # rarity glyph ('M' → '1'). Real MTG sets max out around 4-digit
    # collector numbers. Drop the leading digit.
    if len(collector) >= 5:
        collector = collector[1:]

    return (chosen_set.lower(), str(int(collector)))


# ── top-level entry point ────────────────────────────────────────────────────

def scan_card_image(path: str, query_scryfall: bool = False) -> ScanResult:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    # Downscale large source photos so the min/max area gates match on-device
    long_edge = max(image.shape[:2])
    if long_edge > 1920:
        scale = 1920.0 / long_edge
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    corners, confidence = detect_card_corners(image)
    if corners is None:
        return ScanResult(corners=None, confidence=0.0, rectified=None,
                          bl_text='', name_text='',
                          set_code=None, collector_number=None)

    rectified = rectify(image, corners)

    # Try every crop × OCR variant combination; keep the first parse that
    # succeeds. Fall back to any produced text so bl_text is never empty
    # when OCR actually produced something.
    parsed = None
    bl_text = ''
    for rect in (CROP_BOTTOM_LEFT, CROP_RETRO_BOTTOM):
        cropped = _crop(rectified, rect)
        for txt, _variant in _ocr_variants(cropped):
            if not bl_text and txt:
                bl_text = txt
            p = parse_set_and_number(txt)
            if p is not None:
                parsed = p
                bl_text = txt
                break
        if parsed is not None:
            break
    name_text = run_ocr(_crop(rectified, CROP_NAME))

    set_code, collector_number = (parsed if parsed else (None, None))

    card_name: Optional[str] = None
    if query_scryfall:
        card_name = _scryfall_lookup(set_code, collector_number, name_text)

    return ScanResult(
        corners=corners, confidence=confidence, rectified=rectified,
        bl_text=bl_text, name_text=name_text,
        set_code=set_code, collector_number=collector_number,
        card_name=card_name,
    )


def _scryfall_lookup(set_code, collector, name_text):
    """Resolve to a specific card name via Scryfall.

    Strategy:
      1. /cards/{set}/{collector} for a direct hit when parse succeeded.
      2. Fuzzy named lookup on progressively cleaned substrings of the
         name OCR — sometimes the first line is too noisy but a cleaner
         sub-query hits.
      3. /cards/search ?q=name:{word} fallback that accepts ambiguous
         matches and picks the first result.
    """
    import requests

    def get_json(url, params=None):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    # 1. Direct set+collector lookup
    if set_code and collector:
        j = get_json(f'https://api.scryfall.com/cards/{set_code}/{collector}')
        if j and 'name' in j:
            return j['name']

    if not name_text:
        return None

    # Build cleaned candidate query strings from name OCR
    clean_lines = []
    for raw_line in name_text.splitlines():
        # Strip non-letter noise and runs of single chars
        cleaned = re.sub(r'[^A-Za-z,\' ]', ' ', raw_line).strip()
        # Keep words length >=3 to drop OCR flecks ("i", "a", etc)
        words = [w for w in re.split(r'\s+', cleaned) if len(w) >= 3]
        if words:
            clean_lines.append(' '.join(words))

    tried = set()
    for query in clean_lines:
        if not query or query in tried:
            continue
        tried.add(query)

        # 2. Fuzzy named lookup
        j = get_json('https://api.scryfall.com/cards/named', params={'fuzzy': query})
        if j and 'name' in j:
            return j['name']

        # 3. Search fallback — query unique words against card names and pick the
        # first result. Handles ambiguous fuzzy matches.
        j = get_json('https://api.scryfall.com/cards/search',
                     params={'q': f'name:{query}'})
        if j and j.get('data'):
            return j['data'][0].get('name')

    return None
