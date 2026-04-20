"""Quick inspection helper — prints detection + OCR results per image and
dumps intermediate artifacts (rectified crop, BL/name crops) to /tmp for
visual review."""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from card_pipeline import (
    scan_card_image, CROP_BOTTOM_LEFT, CROP_RETRO_BOTTOM, CROP_NAME,
    detect_card_corners, rectify, _crop,
)

IMAGES = os.path.join(os.path.dirname(__file__), '..', 'images')
OUT = '/tmp/card_pipeline_debug'
os.makedirs(OUT, exist_ok=True)

CASES = [
    'Excava-SOC-0002.png',
    'Gandalf-LTR-0322.jpg',
    'IMG_0446.jpg',
    'IMG_0447.jpg',
    'IMG_0448.jpg',
    'image0.jpg',
]

for name in CASES:
    path = os.path.join(IMAGES, name)
    print(f"\n=== {name} ===")
    img = cv2.imread(path)
    if img is None:
        print(f"  (cannot read)")
        continue

    # Resize as the pipeline does
    long_edge = max(img.shape[:2])
    if long_edge > 1920:
        scale = 1920.0 / long_edge
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"  image size after resize: {img.shape[1]}x{img.shape[0]}")

    stats: dict = {}
    corners, conf = detect_card_corners(img, stats=stats)
    print(f"  stats: {stats}")
    # probe card-ness for best quad per path
    from card_pipeline import _find_best_quad, _build_edge_map_canny, _build_edge_map_threshold, _looks_like_card_border
    gray_dbg2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for label, em in [('canny', _build_edge_map_canny(img)), ('thresh', _build_edge_map_threshold(img))]:
        tmp = {'contours': 0, 'rejected_edge_touch': 0, 'rejected_vertex_count': 0,
               'rejected_rectangularity': 0, 'rejected_area': 0, 'rejected_convex': 0,
               'rejected_angle': 0, 'rejected_ar': 0, 'passed': 0}
        b = _find_best_quad(img, em, tmp)
        if b:
            p, c = b
            mask_outer = __import__('numpy').zeros(gray_dbg2.shape, dtype='uint8')
            cv2.fillPoly(mask_outer, [p.astype('int32')], 255)
            k = max(7, int(min(gray_dbg2.shape) * 0.015))
            if k % 2 == 0: k += 1
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mi = cv2.erode(mask_outer, kern, iterations=1)
            mb = cv2.subtract(mask_outer, mi)
            bmean = cv2.mean(gray_dbg2, mask=mb)[0] if cv2.countNonZero(mb) else -1
            imean = cv2.mean(gray_dbg2, mask=mi)[0] if cv2.countNonZero(mi) else -1
            print(f"  {label}: conf={c:.3f} border={bmean:.0f} interior={imean:.0f} diff={imean-bmean:.0f} card-like={_looks_like_card_border(gray_dbg2, p)}")
        else:
            print(f"  {label}: no candidate")
    # also print top-5 contour areas as fraction of image to diagnose
    gray_dbg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray_dbg)
    filt = cv2.bilateralFilter(eq, 9, 75, 75)
    import numpy as np
    med = float(np.median(filt))
    edges = cv2.Canny(filt, max(0, 0.67 * med), min(255, 1.33 * med))
    k = max(5, int(min(img.shape[1], img.shape[0]) * 0.004))
    if k % 2 == 0: k += 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in cnts], reverse=True)[:5]
    img_area = img.shape[0] * img.shape[1]
    print(f"  top-5 contour area fractions: {[f'{a/img_area:.3f}' for a in areas]}")
    if corners is None:
        print(f"  DETECTION FAILED")
        # Save intermediate edge map for inspection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        filt = cv2.bilateralFilter(eq, 9, 75, 75)
        import numpy as np
        med = float(np.median(filt))
        edges = cv2.Canny(filt, max(0, 0.67 * med), min(255, 1.33 * med))
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        cv2.imwrite(f"{OUT}/{name}.edges.png", edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  contours found: {len(contours)}")
        continue
    print(f"  corners: {corners.tolist()}")
    print(f"  confidence: {conf:.3f}")

    rect = rectify(img, corners)
    cv2.imwrite(f"{OUT}/{name}.rectified.png", rect)
    cv2.imwrite(f"{OUT}/{name}.bl.png", _crop(rect, CROP_BOTTOM_LEFT))
    cv2.imwrite(f"{OUT}/{name}.retro.png", _crop(rect, CROP_RETRO_BOTTOM))
    cv2.imwrite(f"{OUT}/{name}.name.png", _crop(rect, CROP_NAME))

    r = scan_card_image(path)
    print(f"  BL text: {r.bl_text!r}")
    print(f"  NAME text: {r.name_text!r}")
    print(f"  parsed set/collector: {r.set_code}/{r.collector_number}")
