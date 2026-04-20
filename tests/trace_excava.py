"""Detailed trace for Excava — log every contour the Canny path sees."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
from card_pipeline import (
    _build_edge_map_canny, _build_edge_map_threshold,
    MIN_AREA_FRAC, MAX_AREA_FRAC, EDGE_MARGIN_FRAC,
    _sort_corners, _angle_deg,
)

path = os.path.join(os.path.dirname(__file__), '..', 'images', 'Excava-SOC-0002.png')
img = cv2.imread(path)
print(f"image: {img.shape}")

for strategy_name, edges in [('canny', _build_edge_map_canny(img)),
                              ('threshold', _build_edge_map_threshold(img))]:
    print(f"\n--- {strategy_name} ---")
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = img.shape[:2]
    ia = float(w * h)
    edge_margin = max(3, int(min(w, h) * EDGE_MARGIN_FRAC))
    for i, c in enumerate(contours):
        a = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        x, y, bw, bh = cv2.boundingRect(hull)
        perim = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.04 * perim, True)
        touches = (x < edge_margin or y < edge_margin or
                   x + bw > w - edge_margin or y + bh > h - edge_margin)
        print(f"  #{i} area={a:.0f} ({a/ia*100:.1f}%) verts={len(approx)} bbox=({x},{y},{bw},{bh}) touches_edge={touches}")
