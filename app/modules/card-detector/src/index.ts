import { requireNativeModule } from 'expo-modules-core';
import { VisionCameraProxy } from 'react-native-vision-camera';
import type { Frame, FrameProcessorPlugin } from 'react-native-vision-camera';

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

export async function detectCardCorners(imageUri: string): Promise<CardCorners | null> {
  const Native = requireNativeModule('CardDetector');
  const raw: RawCorners | null = await Native.detectCardCorners(imageUri);
  if (!raw) return null;
  return {
    topLeft:     { x: raw.topLeftX,     y: raw.topLeftY     },
    topRight:    { x: raw.topRightX,    y: raw.topRightY    },
    bottomRight: { x: raw.bottomRightX, y: raw.bottomRightY },
    bottomLeft:  { x: raw.bottomLeftX,  y: raw.bottomLeftY  },
  };
}

/** Call once on component mount. Native registers plugin in OnCreate before JS runs. */
export function initCardDetectorPlugin(): FrameProcessorPlugin | null {
  const p = VisionCameraProxy.initFrameProcessorPlugin('detectCardCornersInFrame', {}) ?? null;
  console.log('[CardDetector] plugin:', p == null ? 'NULL - not registered!' : 'OK');
  return p;
}

/** Call inside useFrameProcessor — pass the plugin from initCardDetectorPlugin(). */
export function detectCardCornersInFrame(frame: Frame, plugin: FrameProcessorPlugin): CardCorners | null {
  'worklet';
  const result = plugin.call(frame) as Record<string, number> | null;
  if (!result) return null;
  return {
    topLeft:     { x: result.topLeftX,     y: result.topLeftY },
    topRight:    { x: result.topRightX,    y: result.topRightY },
    bottomRight: { x: result.bottomRightX, y: result.bottomRightY },
    bottomLeft:  { x: result.bottomLeftX,  y: result.bottomLeftY },
  };
}
