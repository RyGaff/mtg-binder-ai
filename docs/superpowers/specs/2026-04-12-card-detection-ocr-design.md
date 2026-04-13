# Card Detection + Two-Strategy OCR Design

**Date:** 2026-04-12

## Overview

Replace the current full-image OCR approach with a pipeline that first detects the card's physical borders in the image using OpenCV, then derives precise crop regions from those corners. Two strategies are tried in order: set/collector-number lookup (bottom-left corner), then fuzzy card-name lookup (top-left name region).

---

## 1. Motivation

Current scanner runs OCR on the full camera frame and parses the last 3 lines for a set code + collector number. This fails when:
- The card occupies only part of the frame (background text pollutes the OCR)
- The card is at an angle (bottom of card is not the last OCR line)
- The bottom-left stamp is too small relative to the image

Cropping to the correct card region before OCR eliminates these failure modes.

---

## 2. Architecture

### New files

| Path | Purpose |
|------|---------|
| `app/modules/card-detector/src/index.ts` | JS API for native module |
| `app/modules/card-detector/cpp/card_detector.h` | Shared C++ algorithm header |
| `app/modules/card-detector/cpp/card_detector.cpp` | Shared C++ algorithm — the ONE implementation |
| `app/modules/card-detector/ios/CardDetectorBridge.mm` | Thin Obj-C++ bridge: decodes image URI → `cv::Mat`, calls C++, returns result to Swift |
| `app/modules/card-detector/ios/CardDetectorModule.swift` | Expo module registration + calls bridge |
| `app/modules/card-detector/ios/CardDetector.podspec` | CocoaPod spec |
| `app/modules/card-detector/android/build.gradle` | Android build config |
| `app/modules/card-detector/android/src/main/cpp/card_detector_jni.cpp` | Thin JNI bridge: decodes image URI → `cv::Mat`, calls shared C++, returns result to Kotlin |
| `app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt` | Expo module registration + calls JNI |

### Modified files

| Path | Change |
|------|--------|
| `app/src/scanner/ocr.ts` | Add `scanCard()` pipeline; keep `parseSetAndNumber` |
| `app/app/(tabs)/scan.tsx` | Replace inline OCR with `scanCard()`; simplify `ScanPhase`; update debug panel |
| `app/package.json` | Add `expo-image-manipulator` |
| `app/ios/Podfile` | Add `OpenCV` pod |
| `app/android/app/build.gradle` | Add OpenCV Maven dependency |

---

## 3. Native Module: `CardDetector`

### JS API (`app/modules/card-detector/src/index.ts`)

```ts
export type Point = { x: number; y: number };  // normalized 0.0–1.0, top-left origin

export type CardCorners = {
  topLeft: Point;
  topRight: Point;
  bottomLeft: Point;
  bottomRight: Point;
};

/**
 * Detects the largest card-shaped rectangle in the image.
 * Returns normalized corner coordinates (top-left origin, 0–1 range),
 * or null if no card-shaped contour is found.
 */
export function detectCardCorners(imageUri: string): Promise<CardCorners | null>;
```

### Shared C++ algorithm (`card_detector.cpp`)

The algorithm lives once. Both platforms compile and link this file — no duplication.

```cpp
// card_detector.h
struct CardCorners {
    float topLeftX,     topLeftY;
    float topRightX,    topRightY;
    float bottomRightX, bottomRightY;
    float bottomLeftX,  bottomLeftY;
};

// Returns false if no card-shaped contour found.
bool detectCardCorners(const cv::Mat& image, CardCorners& out);
```

Pipeline in `card_detector.cpp`:

1. Convert BGR mat to grayscale
2. Gaussian blur: kernel 5×5, sigma 0
3. Canny edge detection: threshold1=50, threshold2=150
4. `findContours` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
5. For each contour, `approxPolyDP` with epsilon = 2% of arc length, closed=true
6. Filter: keep only 4-vertex contours whose area ≥ 15% of total image area
7. Pick the contour with the largest area
8. Sort 4 corners: topLeft (min x+y), topRight (max x−y), bottomRight (max x+y), bottomLeft (max y−x)
9. Normalize to 0–1: divide x by `image.cols`, y by `image.rows`
10. Return false if no qualifying contour found

### iOS bridge (`CardDetectorBridge.mm`)

~30 lines of Objective-C++:
- Accepts a file URI string
- Loads image with `cv::imread` (OpenCV handles file:// paths after stripping prefix)
- Calls `detectCardCorners(mat, corners)`
- Returns an NSDictionary of the 8 normalized floats to Swift

### Android JNI bridge (`card_detector_jni.cpp`)

~30 lines of C++ with JNI signatures:
- Accepts a file path Java string
- Loads image with `cv::imread`
- Calls `detectCardCorners(mat, corners)`
- Returns a `jfloatArray` of the 8 normalized floats to Kotlin

### iOS dependencies

```ruby
# app/ios/Podfile
pod 'OpenCV', '~> 4.9.0'
```

### Android dependencies

```gradle
// app/android/app/build.gradle
implementation 'org.opencv:opencv:4.9.0'

// CMakeLists.txt links card_detector.cpp + card_detector_jni.cpp
externalNativeBuild { cmake { path "src/main/cpp/CMakeLists.txt" } }
```

### Coordinate system

Both bridges decode images using `cv::imread` which uses top-left origin on both platforms. No Y-axis flip needed. Coordinates returned are directly normalized.

---

## 4. JS Pipeline: `scanCard` in `ocr.ts`

### New exports

```ts
export type ScanResult =
  | { strategy: 'set_number'; card: CachedCard }
  | { strategy: 'name';       card: CachedCard };

export async function scanCard(uri: string): Promise<ScanResult>
```

### `parseSetAndNumber` — unchanged

Existing pure function stays as-is. No behavior change.

### Pipeline steps

```
1. detectCardCorners(uri)
     └─ null → throw new Error('No card detected in image')
     └─ corners →

2. Get image dimensions:
     const { width: imgW, height: imgH } =
       await ImageManipulator.manipulateAsync(uri, [])
     // manipulateAsync with empty actions returns { uri, width, height }

3. Compute card pixel dimensions from corners:
     cardWidthPx  = distance(bottomLeft, bottomRight) * imgW
     cardHeightPx = distance(topLeft, bottomLeft) * imgH
     (Euclidean distance on normalized coords, then scale)

4. Strategy 1 — Set/Number (bottom-left crop):
     crop = {
       originX: bottomLeft.x * imgW,
       originY: (bottomLeft.y * imgH) - (0.15 * cardHeightPx),
       width:   0.25 * cardWidthPx,
       height:  0.15 * cardHeightPx,
     }
     clamp all values to image bounds before passing to ImageManipulator
     croppedUri = await ImageManipulator.manipulateAsync(uri, [{ crop }])
     ocrText    = await runOcr(croppedUri)
     parsed     = parseSetAndNumber(ocrText)
     if (parsed):
       try:
         card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber)
         return { strategy: 'set_number', card }
       catch: // Scryfall 404 or network error — fall through to Strategy 2

5. Strategy 2 — Name (top-left crop):
     crop = {
       originX: topLeft.x * imgW,
       originY: topLeft.y * imgH,
       width:   0.65 * cardWidthPx,
       height:  0.12 * cardHeightPx,
     }
     clamp all values to image bounds
     croppedUri = await ImageManipulator.manipulateAsync(uri, [{ crop }])
     ocrText    = await runOcr(croppedUri)
     nameLine   = first non-empty, non-numeric line of ocrText
     if (!nameLine) throw new Error('No text found in name region')
     card = await fetchCardByName(nameLine.trim())
     return { strategy: 'name', card }
```

### `runOcr` helper — stays in `ocr.ts`

Move the inline `runOcr` from `scan.tsx` to `ocr.ts` as a module-level async function (not exported — internal to the pipeline). Same implementation: `resolveToFileUri` → `TextRecognition.recognize`.

`resolveToFileUri` also moves to `ocr.ts`.

---

## 5. `scan.tsx` Changes

### `ScanPhase` — simplified

Remove `ocr_raw` and `parsed` states. New union:

```ts
type ScanPhase =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'fetching' }
  | { status: 'error'; message: string };
```

### `processUri` — simplified

```ts
const processUri = useCallback(async (uri: string, reschedule: boolean) => {
  try {
    setPhase({ status: 'scanning' });
    const result = await scanCard(uri);
    setScanStrategy(result.strategy);   // local state for debug panel
    setPhase({ status: 'fetching' });
    upsertCard(result.card);
    addRecentScan(result.card);
    setLastScannedId(result.card.scryfall_id);
    setPhase({ status: 'idle' });
    setSuccessCard(result.card.name);
    successTimerRef.current = setTimeout(() => {
      successTimerRef.current = null;
      setSuccessCard(null);
      setPickedImageUri(null);
      if (reschedule) runScanCycle();
    }, 1500);
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Unknown error';
    setPhase({ status: 'error', message: msg });
    if (reschedule && msg === 'No card detected in image') {
      // Retry silently — no toast for "no card yet"
      scanLoopRef.current = setTimeout(runScanCycle, 1200);
    }
  }
}, [stopScanning, setLastScannedId, addRecentScan]); // eslint-disable-line
```

### Remove from `scan.tsx`

- `runOcr` function (moved to `ocr.ts`)
- `resolveToFileUri` function (moved to `ocr.ts`)
- `setOcrText` state (no longer needed)
- `ocrText` prop on `OcrDebugPanel`

### `OcrDebugPanel` — updated

Replace raw OCR text display with strategy indicator:

```
OCR DEBUG
strategy: SET_NUMBER   ← or NAME
SET · #042             ← or card name candidate
[ok / fail badge]
```

New props: `phase: ScanPhase`, `strategy: 'set_number' | 'name' | null`

### New local state in `ScanScreen`

```ts
const [scanStrategy, setScanStrategy] = useState<'set_number' | 'name' | null>(null);
```

---

## 6. Error Handling Summary

| Error | Source | Behavior |
|-------|--------|----------|
| `"No card detected in image"` | OpenCV found no rectangle | Silent retry (live camera) or error toast (library pick) |
| `"No text found in name region"` | Name crop OCR empty | Error toast |
| Scryfall 404 (set/number) | Bad parse or unknown set | Fall through to Strategy 2 |
| Scryfall 404 (name) | Card name not recognized | Error toast |

Note: Scryfall 404 on Strategy 1 falls through to Strategy 2 rather than surfacing as an error. The existing `fetchCardBySetNumber` throws on non-OK responses — wrap Strategy 1's fetch in a try/catch that falls through instead of re-throwing.

---

## 7. Out of Scope

- Perspective correction (de-skewing rotated cards)
- OCR preprocessing (grayscale/threshold) on the crop before text recognition — Vision framework / ML Kit handle this internally
- Caching detected corners across scan cycles
