# Card Detection + Two-Strategy OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-image OCR with a pipeline that detects the card's physical borders using OpenCV (shared C++), crops to the bottom-left for set/number, falls back to top-left name region, and fuzzy-matches via Scryfall.

**Architecture:** A local Expo native module (`card-detector`) wraps a single C++ algorithm compiled on both platforms. Thin Obj-C++ (iOS) and JNI (Android) bridges load the image and call the shared code. `ocr.ts` orchestrates: detect corners → crop → OCR → Scryfall. `scan.tsx` calls `scanCard()` instead of inline OCR.

**Tech Stack:** OpenCV 4.9 (C++), Expo Modules API, Swift (iOS bridge), Kotlin + JNI (Android bridge), expo-image-manipulator, react-native-text-recognition

---

## File Map

| File | Action |
|------|--------|
| `app/modules/card-detector/package.json` | Create — module manifest |
| `app/modules/card-detector/cpp/card_detector.h` | Create — C++ API |
| `app/modules/card-detector/cpp/card_detector.cpp` | Create — shared algorithm |
| `app/modules/card-detector/ios/CardDetector.podspec` | Create — CocoaPod spec |
| `app/modules/card-detector/ios/CardDetectorBridge.h` | Create — Obj-C++ header |
| `app/modules/card-detector/ios/CardDetectorBridge.mm` | Create — iOS thin bridge |
| `app/modules/card-detector/ios/CardDetectorModule.swift` | Create — Expo module registration |
| `app/modules/card-detector/android/build.gradle` | Create — Android build config |
| `app/modules/card-detector/android/src/main/cpp/CMakeLists.txt` | Create — CMake config |
| `app/modules/card-detector/android/src/main/cpp/card_detector_jni.cpp` | Create — Android thin bridge |
| `app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt` | Create — Android Expo module |
| `app/modules/card-detector/src/index.ts` | Create — JS API |
| `app/package.json` | Modify — add card-detector + expo-image-manipulator |
| `app/src/scanner/ocr.ts` | Modify — add scanCard() pipeline, move runOcr/resolveToFileUri from scan.tsx |
| `app/__tests__/scanner/ocr.test.ts` | Modify — add scanCard() tests |
| `app/app/(tabs)/scan.tsx` | Modify — use scanCard(), simplify ScanPhase, update OcrDebugPanel |

---

### Task 1: Module scaffold + dependencies

**Files:**
- Create: `app/modules/card-detector/package.json`
- Modify: `app/package.json`

- [ ] **Step 1: Create module package.json**

```json
// app/modules/card-detector/package.json
{
  "name": "card-detector",
  "version": "1.0.0",
  "description": "OpenCV card corner detection",
  "main": "src/index",
  "peerDependencies": {
    "expo-modules-core": "*"
  }
}
```

- [ ] **Step 2: Register local module and add expo-image-manipulator in app/package.json**

In `app/package.json`, add to `"dependencies"`:

```json
"card-detector": "file:modules/card-detector",
"expo-image-manipulator": "~13.0.6"
```

- [ ] **Step 3: Install**

```bash
cd app && npm install
```

Expected: `node_modules/card-detector` symlink created, `expo-image-manipulator` installed.

- [ ] **Step 4: Verify autolinking will find the module**

```bash
cd app && npx expo-modules-autolinking search 2>/dev/null | grep -i card || echo "autolinking check done"
```

(Output may vary — the module will be discovered at native build time via `use_expo_modules!`)

- [ ] **Step 5: Commit**

```bash
cd app && git add package.json package-lock.json modules/card-detector/package.json
git commit -m "chore: add card-detector local module scaffold and expo-image-manipulator"
```

---

### Task 2: Shared C++ algorithm

**Files:**
- Create: `app/modules/card-detector/cpp/card_detector.h`
- Create: `app/modules/card-detector/cpp/card_detector.cpp`

No Jest tests for C++ — algorithm correctness verified through integration (Tasks 3–4). Unit tests for the JS pipeline in Task 5 mock `detectCardCorners` directly.

- [ ] **Step 1: Create header**

```cpp
// app/modules/card-detector/cpp/card_detector.h
#pragma once
#include <opencv2/opencv.hpp>

struct CardCorners {
    float topLeftX,     topLeftY;
    float topRightX,    topRightY;
    float bottomRightX, bottomRightY;
    float bottomLeftX,  bottomLeftY;
};

/**
 * Detects the largest card-shaped rectangle in the image.
 * Corners are normalized to 0–1 range (top-left origin).
 * Returns false if no qualifying 4-vertex contour found.
 */
bool detectCardCorners(const cv::Mat& image, CardCorners& out);
```

- [ ] **Step 2: Create implementation**

```cpp
// app/modules/card-detector/cpp/card_detector.cpp
#include "card_detector.h"
#include <algorithm>
#include <vector>

bool detectCardCorners(const cv::Mat& image, CardCorners& out) {
    if (image.empty()) return false;

    cv::Mat gray, blurred, edges;

    // Grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Gaussian blur
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Canny edge detection
    cv::Canny(blurred, edges, 50, 150);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double imageArea = static_cast<double>(image.cols) * image.rows;
    double minArea = imageArea * 0.15;

    std::vector<cv::Point> best;
    double bestArea = 0;

    for (const auto& contour : contours) {
        double perimeter = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        if (approx.size() != 4) continue;

        double area = cv::contourArea(approx);
        if (area < minArea) continue;

        if (area > bestArea) {
            bestArea = area;
            best = approx;
        }
    }

    if (best.empty()) return false;

    // Sort by x+y sum: index 0 = topLeft, index 3 = bottomRight
    std::sort(best.begin(), best.end(), [](const cv::Point& a, const cv::Point& b) {
        return (a.x + a.y) < (b.x + b.y);
    });

    cv::Point topLeft     = best[0];
    cv::Point bottomRight = best[3];

    // Of the two middle points, topRight has greater (x - y)
    cv::Point topRight, bottomLeft;
    if ((best[1].x - best[1].y) > (best[2].x - best[2].y)) {
        topRight   = best[1];
        bottomLeft = best[2];
    } else {
        topRight   = best[2];
        bottomLeft = best[1];
    }

    float w = static_cast<float>(image.cols);
    float h = static_cast<float>(image.rows);

    out.topLeftX     = topLeft.x     / w;  out.topLeftY     = topLeft.y     / h;
    out.topRightX    = topRight.x    / w;  out.topRightY    = topRight.y    / h;
    out.bottomRightX = bottomRight.x / w;  out.bottomRightY = bottomRight.y / h;
    out.bottomLeftX  = bottomLeft.x  / w;  out.bottomLeftY  = bottomLeft.y  / h;

    return true;
}
```

- [ ] **Step 3: Commit**

```bash
git add app/modules/card-detector/cpp/
git commit -m "feat: add shared C++ card corner detection algorithm"
```

---

### Task 3: iOS bridge + Expo module

**Files:**
- Create: `app/modules/card-detector/ios/CardDetector.podspec`
- Create: `app/modules/card-detector/ios/CardDetectorBridge.h`
- Create: `app/modules/card-detector/ios/CardDetectorBridge.mm`
- Create: `app/modules/card-detector/ios/CardDetectorModule.swift`

No Jest tests — native code. Verified by running the app on device in Task 4 testing.

- [ ] **Step 1: Create podspec**

```ruby
# app/modules/card-detector/ios/CardDetector.podspec
require 'json'
package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'CardDetector'
  s.version        = package['version']
  s.summary        = 'OpenCV card corner detection'
  s.homepage       = 'https://github.com/RyGaff/mtg-binder-ai'
  s.license        = 'MIT'
  s.authors        = { 'RyGaff' => '' }
  s.platform       = :ios, '15.1'
  s.source         = { :path => '.' }
  # Include Swift/ObjC module files and the shared C++ sources
  s.source_files   = 'ios/**/*.{h,m,mm,swift}', '../cpp/**/*.{h,cpp}'
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY'           => 'libc++',
  }
  s.dependency 'OpenCV', '~> 4.9.0'
  s.dependency 'ExpoModulesCore'
end
```

- [ ] **Step 2: Create Obj-C++ bridge header**

```objc
// app/modules/card-detector/ios/CardDetectorBridge.h
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CardDetectorBridge : NSObject
/**
 * Detects card corners in the image at the given file URI.
 * Returns a dictionary with keys topLeftX/Y, topRightX/Y,
 * bottomRightX/Y, bottomLeftX/Y (all NSNumber doubles, 0–1 range),
 * or nil if no card was found.
 */
+ (NSDictionary<NSString *, NSNumber *> * _Nullable)detectCornersFromURI:(NSString *)uri;
@end

NS_ASSUME_NONNULL_END
```

- [ ] **Step 3: Create Obj-C++ bridge implementation**

```objc
// app/modules/card-detector/ios/CardDetectorBridge.mm
#import "CardDetectorBridge.h"
#import "card_detector.h"
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>

@implementation CardDetectorBridge

+ (NSDictionary<NSString *, NSNumber *> *)detectCornersFromURI:(NSString *)uri {
    // Strip file:// prefix
    NSString *path = [uri hasPrefix:@"file://"]
        ? [uri substringFromIndex:7]
        : uri;

    UIImage *image = [UIImage imageWithContentsOfFile:path];
    if (!image) return nil;

    cv::Mat mat;
    UIImageToMat(image, mat);
    if (mat.empty()) return nil;

    // UIImageToMat produces RGBA — convert to BGR for OpenCV
    if (mat.channels() == 4) {
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    }

    CardCorners corners;
    if (!detectCardCorners(mat, corners)) return nil;

    return @{
        @"topLeftX":     @(corners.topLeftX),
        @"topLeftY":     @(corners.topLeftY),
        @"topRightX":    @(corners.topRightX),
        @"topRightY":    @(corners.topRightY),
        @"bottomRightX": @(corners.bottomRightX),
        @"bottomRightY": @(corners.bottomRightY),
        @"bottomLeftX":  @(corners.bottomLeftX),
        @"bottomLeftY":  @(corners.bottomLeftY),
    };
}

@end
```

- [ ] **Step 4: Create Swift Expo module**

```swift
// app/modules/card-detector/ios/CardDetectorModule.swift
import ExpoModulesCore

public class CardDetectorModule: Module {
    public func definition() -> ModuleDefinition {
        Name("CardDetector")

        AsyncFunction("detectCardCorners") { (uri: String) -> [String: Double]? in
            guard let raw = CardDetectorBridge.detectCorners(fromURI: uri) else {
                return nil
            }
            // NSDictionary<String, NSNumber> → [String: Double]
            return raw.reduce(into: [String: Double]()) { dict, pair in
                if let key = pair.key as? String, let val = pair.value as? Double {
                    dict[key] = val
                }
            }
        }
    }
}
```

- [ ] **Step 5: Rebuild iOS**

```bash
cd app && npx expo run:ios
```

Expected: Compiles successfully. OpenCV pod installs. CardDetector module linked.

If pod install fails for OpenCV version:
```bash
cd app/ios && pod install --repo-update
```

- [ ] **Step 6: Commit**

```bash
git add app/modules/card-detector/ios/
git commit -m "feat: add iOS bridge for card corner detection (OpenCV + Expo module)"
```

---

### Task 4: Android bridge + Expo module

**Prerequisite:** Android project must exist. If `app/android/` is missing, run:
```bash
cd app && npx expo prebuild --platform android
```

**Files:**
- Create: `app/modules/card-detector/android/build.gradle`
- Create: `app/modules/card-detector/android/src/main/cpp/CMakeLists.txt`
- Create: `app/modules/card-detector/android/src/main/cpp/card_detector_jni.cpp`
- Create: `app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt`

- [ ] **Step 1: Create Android build.gradle**

```gradle
// app/modules/card-detector/android/build.gradle
apply plugin: 'com.android.library'
apply plugin: 'kotlin-android'
apply plugin: 'expo-module'

android {
    compileSdk 35
    namespace 'expo.modules.carddetector'

    defaultConfig {
        minSdk 24
        externalNativeBuild {
            cmake {
                cppFlags "-std=c++17"
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }

    buildFeatures {
        prefab true   // Required for OpenCV Maven AAR C++ headers
    }

    externalNativeBuild {
        cmake {
            path "src/main/cpp/CMakeLists.txt"
            version "3.22.1"
        }
    }
}

dependencies {
    implementation 'org.opencv:opencv:4.9.0'
    implementation "org.jetbrains.kotlin:kotlin-stdlib-jdk7:$kotlin_version"
}
```

- [ ] **Step 2: Create CMakeLists.txt**

```cmake
# app/modules/card-detector/android/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)
project(card_detector)

# OpenCV via prefab (from Maven AAR)
find_package(opencv REQUIRED CONFIG)

add_library(
    card_detector
    SHARED
    card_detector_jni.cpp
    # Shared C++ algorithm (relative path from this CMakeLists location)
    ../../../../../cpp/card_detector.cpp
)

target_include_directories(card_detector PRIVATE
    ../../../../../cpp
)

target_link_libraries(card_detector
    opencv::opencv_core
    opencv::opencv_imgproc
    opencv::opencv_imgcodecs
    android
    log
)
```

- [ ] **Step 3: Create JNI bridge**

```cpp
// app/modules/card-detector/android/src/main/cpp/card_detector_jni.cpp
#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "card_detector.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_expo_modules_carddetector_CardDetectorModule_detectCornersNative(
        JNIEnv *env, jobject /* this */, jstring filePath) {

    const char *path = env->GetStringUTFChars(filePath, nullptr);
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    env->ReleaseStringUTFChars(filePath, path);

    if (image.empty()) return nullptr;

    CardCorners corners;
    if (!detectCardCorners(image, corners)) return nullptr;

    jfloatArray result = env->NewFloatArray(8);
    float data[8] = {
        corners.topLeftX,     corners.topLeftY,
        corners.topRightX,    corners.topRightY,
        corners.bottomRightX, corners.bottomRightY,
        corners.bottomLeftX,  corners.bottomLeftY,
    };
    env->SetFloatArrayRegion(result, 0, 8, data);
    return result;
}
```

- [ ] **Step 4: Create Kotlin Expo module**

```kotlin
// app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt
package expo.modules.carddetector

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class CardDetectorModule : Module() {

    companion object {
        init {
            System.loadLibrary("card_detector")
        }
    }

    private external fun detectCornersNative(filePath: String): FloatArray?

    override fun definition() = ModuleDefinition {
        Name("CardDetector")

        AsyncFunction("detectCardCorners") { uri: String ->
            val path = if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
            val raw = detectCornersNative(path) ?: return@AsyncFunction null
            if (raw.size != 8) return@AsyncFunction null
            mapOf(
                "topLeftX"     to raw[0], "topLeftY"     to raw[1],
                "topRightX"    to raw[2], "topRightY"    to raw[3],
                "bottomRightX" to raw[4], "bottomRightY" to raw[5],
                "bottomLeftX"  to raw[6], "bottomLeftY"  to raw[7],
            )
        }
    }
}
```

- [ ] **Step 5: Rebuild Android**

```bash
cd app && npx expo run:android
```

Expected: Compiles successfully. OpenCV AAR downloaded. card_detector.so built. CardDetector module linked.

- [ ] **Step 6: Commit**

```bash
git add app/modules/card-detector/android/
git commit -m "feat: add Android bridge for card corner detection (OpenCV + JNI + Expo module)"
```

---

### Task 5: JS module API

**Files:**
- Create: `app/modules/card-detector/src/index.ts`

- [ ] **Step 1: Create the JS API**

```ts
// app/modules/card-detector/src/index.ts
import { requireNativeModule } from 'expo-modules-core';

const Native = requireNativeModule('CardDetector');

export type Point = { x: number; y: number }; // normalized 0–1, top-left origin

export type CardCorners = {
  topLeft:     Point;
  topRight:    Point;
  bottomRight: Point;
  bottomLeft:  Point;
};

type RawCorners = {
  topLeftX:     number; topLeftY:     number;
  topRightX:    number; topRightY:    number;
  bottomRightX: number; bottomRightY: number;
  bottomLeftX:  number; bottomLeftY:  number;
};

/**
 * Detects the largest card-shaped rectangle in the image at the given URI.
 * Returns normalized corner coordinates (top-left origin, 0–1 range),
 * or null if no card-shaped contour was found.
 */
export async function detectCardCorners(imageUri: string): Promise<CardCorners | null> {
  const raw: RawCorners | null = await Native.detectCardCorners(imageUri);
  if (!raw) return null;
  return {
    topLeft:     { x: raw.topLeftX,     y: raw.topLeftY     },
    topRight:    { x: raw.topRightX,    y: raw.topRightY    },
    bottomRight: { x: raw.bottomRightX, y: raw.bottomRightY },
    bottomLeft:  { x: raw.bottomLeftX,  y: raw.bottomLeftY  },
  };
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd app && npx tsc --noEmit 2>&1 | grep -v "__tests__"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add app/modules/card-detector/src/
git commit -m "feat: add JS API for card-detector native module"
```

---

### Task 6: Update ocr.ts — scanCard pipeline

**Files:**
- Modify: `app/src/scanner/ocr.ts`
- Modify: `app/__tests__/scanner/ocr.test.ts`

- [ ] **Step 1: Write failing tests for scanCard**

Replace the contents of `app/__tests__/scanner/ocr.test.ts` with:

```ts
// Mocks must be declared before imports
jest.mock('../../modules/card-detector/src', () => ({
  detectCardCorners: jest.fn(),
}));

jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn(),
}));

jest.mock('react-native-text-recognition', () => ({
  default: { recognize: jest.fn() },
}));

jest.mock('expo-file-system', () => ({
  File: jest.fn().mockImplementation((p: string) => ({ uri: `file://${p}`, copy: jest.fn() })),
  Paths: { cache: '/cache' },
}));

jest.mock('../../src/api/scryfall', () => ({
  fetchCardBySetNumber: jest.fn(),
  fetchCardByName: jest.fn(),
}));

import { parseSetAndNumber, scanCard } from '../../src/scanner/ocr';
import { detectCardCorners } from '../../modules/card-detector/src';
import * as ImageManipulator from 'expo-image-manipulator';
import { fetchCardBySetNumber, fetchCardByName } from '../../src/api/scryfall';
import TextRecognition from 'react-native-text-recognition';

const mockDetect = detectCardCorners as jest.Mock;
const mockManipulate = ImageManipulator.manipulateAsync as jest.Mock;
const mockRecognize = (TextRecognition as any).recognize as jest.Mock;
const mockFetchBySet = fetchCardBySetNumber as jest.Mock;
const mockFetchByName = fetchCardByName as jest.Mock;

const CORNERS = {
  topLeft:     { x: 0.1, y: 0.1 },
  topRight:    { x: 0.9, y: 0.1 },
  bottomRight: { x: 0.9, y: 0.9 },
  bottomLeft:  { x: 0.1, y: 0.9 },
};

const MOCK_CARD = {
  scryfall_id: 'abc-123',
  name: 'Lightning Bolt',
  set_code: 'lea',
  collector_number: '161',
  mana_cost: '{R}',
  type_line: 'Instant',
  oracle_text: 'Deal 3 damage.',
  color_identity: '["R"]',
  image_uri: 'https://example.com/bolt.jpg',
  prices: '{"usd":"1.00"}',
  keywords: '[]',
  cached_at: Date.now(),
};

// Stub manipulateAsync to return a fake URI + dimensions
const stubManipulate = (width = 1000, height = 1400) => {
  mockManipulate.mockResolvedValue({ uri: 'file:///crop.jpg', width, height });
};

beforeEach(() => {
  jest.clearAllMocks();
  stubManipulate();
});

// ── parseSetAndNumber (existing tests kept) ──────────────────────────────────

describe('parseSetAndNumber', () => {
  it('parses set-code-first format', () => {
    expect(parseSetAndNumber('lea 161/302')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('parses set code without slash', () => {
    expect(parseSetAndNumber('m21 420')).toEqual({ setCode: 'm21', collectorNumber: '420' });
  });
  it('parses 3-char set codes', () => {
    expect(parseSetAndNumber('cmr 085/361')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });
  it('handles mixed case', () => {
    expect(parseSetAndNumber('LEA 161')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('parses modern card format: number rarity set lang', () => {
    expect(parseSetAndNumber('042/350 R IKO EN')).toEqual({ setCode: 'iko', collectorNumber: '042' });
  });
  it('parses modern format without total', () => {
    expect(parseSetAndNumber('161 R LEA EN')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('ignores language and rarity tokens', () => {
    expect(parseSetAndNumber('085/361 C CMR EN')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });
  it('handles OCR noise around the real data', () => {
    expect(parseSetAndNumber('Illustrated by John Avon\n085/361 C CMR EN')).toEqual({
      setCode: 'cmr', collectorNumber: '085',
    });
  });
  it('returns null for unrecognizable text', () => {
    expect(parseSetAndNumber('not a card')).toBeNull();
  });
  it('returns null when no collector number present', () => {
    expect(parseSetAndNumber('IKO EN R')).toBeNull();
  });
});

// ── scanCard ─────────────────────────────────────────────────────────────────

describe('scanCard', () => {
  it('throws when no card detected', async () => {
    mockDetect.mockResolvedValue(null);
    await expect(scanCard('file:///photo.jpg')).rejects.toThrow('No card detected in image');
  });

  it('returns set_number result when bottom-left OCR succeeds', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize.mockResolvedValue(['161 R LEA EN']);
    mockFetchBySet.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('set_number');
    expect(result.card.name).toBe('Lightning Bolt');
    expect(mockFetchBySet).toHaveBeenCalledWith('lea', '161');
    expect(mockFetchByName).not.toHaveBeenCalled();
  });

  it('falls through to name strategy when set/number parse fails', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    // First OCR call (bottom-left) returns no parseable set/number
    // Second OCR call (name) returns the card name
    mockRecognize
      .mockResolvedValueOnce(['not a card'])
      .mockResolvedValueOnce(['Lightning Bolt']);
    mockFetchByName.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('name');
    expect(result.card.name).toBe('Lightning Bolt');
    expect(mockFetchByName).toHaveBeenCalledWith('Lightning Bolt');
  });

  it('falls through to name strategy when Scryfall 404 on set/number', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize
      .mockResolvedValueOnce(['161 R LEA EN'])
      .mockResolvedValueOnce(['Lightning Bolt']);
    mockFetchBySet.mockRejectedValue(new Error('Scryfall 404'));
    mockFetchByName.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('name');
    expect(mockFetchByName).toHaveBeenCalledWith('Lightning Bolt');
  });

  it('throws when name region OCR returns no text', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize.mockResolvedValue([]);

    await expect(scanCard('file:///photo.jpg')).rejects.toThrow('No text found in name region');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd app && npx jest --watchAll=false __tests__/scanner/ocr.test.ts
```

Expected: FAIL — `scanCard` is not exported from `ocr.ts`.

- [ ] **Step 3: Implement scanCard in ocr.ts**

Replace the entire contents of `app/src/scanner/ocr.ts` with:

```ts
import * as ImageManipulator from 'expo-image-manipulator';
import { File, Paths } from 'expo-file-system';
import { detectCardCorners } from '../../modules/card-detector/src';
import { fetchCardBySetNumber, fetchCardByName } from '../api/scryfall';
import type { CachedCard } from '../db/cards';

// ── Types ────────────────────────────────────────────────────────────────────

export type ParsedCard = { setCode: string; collectorNumber: string };

export type ScanResult =
  | { strategy: 'set_number'; card: CachedCard }
  | { strategy: 'name';       card: CachedCard };

// ── Helpers (internal) ───────────────────────────────────────────────────────

const SKIP_TOKENS = new Set([
  'EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'JA', 'KO', 'RU', 'ZH', 'PH', 'CS',
  'R', 'U', 'C', 'M', 'S', 'T', 'L',
  'THE', 'AND', 'FOR', 'YOU', 'MAY', 'TAP', 'PUT', 'TOP', 'NEW',
  'COPY', 'CAST', 'EACH', 'FROM', 'YOUR', 'BEEN', 'THAT', 'CARD',
  'WITH', 'LESS', 'MANA', 'AURA', 'NON',
  'MEE', 'LLC', 'INC', 'LTD', 'ALL', 'TM',
]);

async function resolveToFileUri(uri: string): Promise<string> {
  if (uri.startsWith('file://') || uri.startsWith('/')) return uri;
  const dest = new File(Paths.cache, `scan_ocr_${Date.now()}.jpg`);
  const source = new File(uri);
  source.copy(dest); // sync: expo-file-system File API
  return dest.uri;
}

async function runOcr(uri: string): Promise<string> {
  const TextRecognition = require('react-native-text-recognition').default;
  if (!TextRecognition || typeof TextRecognition.recognize !== 'function') {
    throw new Error(
      'OCR module not available. Run `expo run:ios` to link native dependencies.'
    );
  }
  const resolvedUri = await resolveToFileUri(uri);
  const lines: string[] = await TextRecognition.recognize(resolvedUri);
  return lines.join('\n');
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ── Public: parseSetAndNumber ────────────────────────────────────────────────

/**
 * Parses raw OCR text from the bottom-left corner of an MTG card.
 * Modern format: "042/350 R IKO EN"
 * Older format:  "IKO 042/350"
 */
export function parseSetAndNumber(text: string): ParsedCard | null {
  const upper = text.toUpperCase();

  const numMatch = upper.match(/\b(\d{1,4})(?:\/\d+)?\b/);
  if (!numMatch) return null;
  const collectorNumber = numMatch[1].replace(/^0+(\d)/, '$1');

  const lines = upper.split('\n');
  const bottomLines = lines.slice(-3).join('\n');

  const setCandidates = [...bottomLines.matchAll(/\b([A-Z][A-Z0-9]{1,3})\b/g)]
    .map(m => m[1])
    .filter(s => !SKIP_TOKENS.has(s) && !/^\d+$/.test(s));

  if (setCandidates.length === 0) return null;

  const setCode = (setCandidates.find(s => /^[A-Z]{3}$/.test(s)) ?? setCandidates[0]).toLowerCase();
  return { setCode, collectorNumber };
}

// ── Public: scanCard ─────────────────────────────────────────────────────────

/**
 * Full scanning pipeline:
 * 1. Detect card corners with OpenCV
 * 2. Crop bottom-left → OCR → parse set/number → Scryfall lookup
 * 3. Fallback: crop name region → OCR → fuzzy name lookup
 */
export async function scanCard(uri: string): Promise<ScanResult> {
  const corners = await detectCardCorners(uri);
  if (!corners) throw new Error('No card detected in image');

  // Get image dimensions (manipulateAsync with no actions returns { uri, width, height })
  const info = await ImageManipulator.manipulateAsync(uri, []);
  const imgW = info.width;
  const imgH = info.height;

  // Card dimensions in pixels (Euclidean distance between corners)
  const cardWidthPx = Math.sqrt(
    Math.pow((corners.bottomRight.x - corners.bottomLeft.x) * imgW, 2) +
    Math.pow((corners.bottomRight.y - corners.bottomLeft.y) * imgH, 2)
  );
  const cardHeightPx = Math.sqrt(
    Math.pow((corners.bottomLeft.x - corners.topLeft.x) * imgW, 2) +
    Math.pow((corners.bottomLeft.y - corners.topLeft.y) * imgH, 2)
  );

  // ── Strategy 1: bottom-left crop (set code + collector number) ──

  const blOriginX = clamp(Math.floor(corners.bottomLeft.x * imgW), 0, imgW - 1);
  const blOriginY = clamp(Math.floor(corners.bottomLeft.y * imgH - 0.15 * cardHeightPx), 0, imgH - 1);
  const blWidth   = clamp(Math.ceil(0.25 * cardWidthPx),  1, imgW - blOriginX);
  const blHeight  = clamp(Math.ceil(0.15 * cardHeightPx), 1, imgH - blOriginY);

  const blCrop = await ImageManipulator.manipulateAsync(uri, [
    { crop: { originX: blOriginX, originY: blOriginY, width: blWidth, height: blHeight } },
  ]);
  const blText = await runOcr(blCrop.uri);
  const parsed = parseSetAndNumber(blText);

  if (parsed) {
    try {
      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      return { strategy: 'set_number', card };
    } catch {
      // Scryfall 404 or network error — fall through to name strategy
    }
  }

  // ── Strategy 2: name crop (top-left region) ──

  const tlOriginX = clamp(Math.floor(corners.topLeft.x * imgW), 0, imgW - 1);
  const tlOriginY = clamp(Math.floor(corners.topLeft.y * imgH), 0, imgH - 1);
  const tlWidth   = clamp(Math.ceil(0.65 * cardWidthPx),  1, imgW - tlOriginX);
  const tlHeight  = clamp(Math.ceil(0.12 * cardHeightPx), 1, imgH - tlOriginY);

  const tlCrop = await ImageManipulator.manipulateAsync(uri, [
    { crop: { originX: tlOriginX, originY: tlOriginY, width: tlWidth, height: tlHeight } },
  ]);
  const tlText = await runOcr(tlCrop.uri);

  const nameLine = tlText
    .split('\n')
    .find(l => l.trim().length > 0 && !/^\d+$/.test(l.trim()));

  if (!nameLine) throw new Error('No text found in name region');

  const card = await fetchCardByName(nameLine.trim());
  return { strategy: 'name', card };
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd app && npx jest --watchAll=false __tests__/scanner/ocr.test.ts
```

Expected: all tests PASS (10 parseSetAndNumber + 5 scanCard = 15 tests).

- [ ] **Step 5: Verify TypeScript**

```bash
cd app && npx tsc --noEmit 2>&1 | grep -v "__tests__"
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
cd app && git add src/scanner/ocr.ts __tests__/scanner/ocr.test.ts
git commit -m "feat: add scanCard() pipeline with OpenCV corner detection and two-strategy OCR"
```

---

### Task 7: Update scan.tsx

**Files:**
- Modify: `app/app/(tabs)/scan.tsx`

No unit tests — UI changes verified manually on device.

- [ ] **Step 1: Update imports**

At the top of `app/app/(tabs)/scan.tsx`, replace:

```ts
import { parseSetAndNumber } from '../../src/scanner/ocr';
import { fetchCardBySetNumber } from '../../src/api/scryfall';
```

With:

```ts
import { scanCard } from '../../src/scanner/ocr';
```

Also remove the `File` and `Paths` imports from `expo-file-system` if present (they moved to `ocr.ts`).

- [ ] **Step 2: Simplify ScanPhase**

Replace the `ScanPhase` type:

```ts
// Replace:
type ScanPhase =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'ocr_raw'; text: string }
  | { status: 'parsed'; setCode: string; collectorNumber: string; rawText: string }
  | { status: 'fetching'; setCode: string; collectorNumber: string }
  | { status: 'error'; message: string };

// With:
type ScanPhase =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'fetching' }
  | { status: 'error'; message: string };
```

- [ ] **Step 3: Add scanStrategy state, remove ocrText state**

In `ScanScreen`, remove:
```ts
const [ocrText, setOcrText] = useState<string>('');
```

Add:
```ts
const [scanStrategy, setScanStrategy] = useState<'set_number' | 'name' | null>(null);
```

- [ ] **Step 4: Replace processUri body**

Replace the entire `processUri` useCallback with:

```ts
  const processUri = useCallback(async (uri: string, reschedule: boolean) => {
    try {
      setPhase({ status: 'scanning' });
      const result = await scanCard(uri);
      setScanStrategy(result.strategy);
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
      if (reschedule && msg === 'No card detected in image') {
        // Silent retry — card not in frame yet
        scanLoopRef.current = setTimeout(runScanCycle, 1200);
      } else {
        setPhase({ status: 'error', message: msg });
      }
    }
  }, [setLastScannedId, addRecentScan]); // eslint-disable-line react-hooks/exhaustive-deps
```

- [ ] **Step 5: Remove inline runOcr and resolveToFileUri functions**

Delete the `resolveToFileUri` and `runOcr` function definitions from `scan.tsx` (they now live in `ocr.ts`).

- [ ] **Step 6: Update OcrDebugPanel**

Replace the entire `OcrDebugPanel` component with this slimmed-down version:

```tsx
function OcrDebugPanel({
  phase,
  strategy,
}: {
  phase: ScanPhase;
  strategy: 'set_number' | 'name' | null;
}) {
  if (phase.status === 'idle') return null;

  const strategyLabel = strategy === 'set_number'
    ? 'SET + NUMBER'
    : strategy === 'name'
    ? 'NAME FALLBACK'
    : null;

  const statusLabel = (() => {
    switch (phase.status) {
      case 'scanning': return 'Detecting card…';
      case 'fetching': return 'Looking up card…';
      case 'error':    return phase.message;
      default:         return null;
    }
  })();

  return (
    <View style={debugStyles.panel}>
      <Text style={debugStyles.header}>OCR DEBUG</Text>
      {strategyLabel && (
        <Text style={debugStyles.strategy}>{strategyLabel}</Text>
      )}
      {statusLabel && (
        <Text style={[
          debugStyles.status,
          phase.status === 'error' ? debugStyles.statusError : debugStyles.statusOk,
        ]}>
          {statusLabel}
        </Text>
      )}
    </View>
  );
}
```

- [ ] **Step 7: Update OcrDebugPanel call sites**

Find both places where `<OcrDebugPanel` is rendered and update the props:

```tsx
// Replace: <OcrDebugPanel phase={phase} ocrText={ocrText} />
// With:
<OcrDebugPanel phase={phase} strategy={scanStrategy} />
```

- [ ] **Step 8: Update debugStyles for new panel**

In the `debugStyles` StyleSheet, add two new keys (keep existing keys that are still used):

```ts
  strategy: {
    color: '#4ecdc4',
    fontSize: 11,
    fontWeight: '700' as const,
    fontFamily: 'monospace',
    marginBottom: 4,
    letterSpacing: 0.8,
  },
  status: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  statusOk: {
    color: 'rgba(255,255,255,0.6)',
  },
  statusError: {
    color: '#ef5350',
    fontWeight: '600' as const,
  },
```

- [ ] **Step 9: Update statusLabel in ScanScreen**

The `statusLabel` switch in `ScanScreen` references removed states. Replace:

```ts
// Replace the existing statusLabel block:
const statusLabel = (() => {
  switch (phase.status) {
    case 'idle':     return isActive ? 'Point camera at a card' : 'Tap to start scanning';
    case 'scanning': return 'Detecting card…';
    case 'fetching': return 'Looking up card…';
    case 'error':    return null;
  }
})();
```

- [ ] **Step 10: Verify TypeScript**

```bash
cd app && npx tsc --noEmit 2>&1 | grep -v "__tests__"
```

Expected: no output.

- [ ] **Step 11: Commit**

```bash
cd app && git add "app/(tabs)/scan.tsx"
git commit -m "feat: use scanCard() in scan screen, simplify ScanPhase, update debug panel"
```
