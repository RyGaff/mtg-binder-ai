import { requireNativeModule } from 'expo-modules-core';
import { VisionCameraProxy } from 'react-native-vision-camera';
import type { Frame } from 'react-native-vision-camera';

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

const plugin = VisionCameraProxy.initFrameProcessorPlugin('detectCardCornersInFrame', {});

/**
 * Frame processor plugin for react-native-vision-camera v4.
 * Runs on raw camera frames at native speed (~30 fps).
 * Must be called inside a `useFrameProcessor` worklet.
 * Returns normalized corner coordinates or null if no card detected.
 */
export function detectCardCornersInFrame(frame: Frame): CardCorners | null {
  'worklet';
  if (plugin == null) return null;
  const result = plugin.call(frame) as Record<string, number> | null;
  if (!result) return null;
  return {
    topLeft:     { x: result.topLeftX,     y: result.topLeftY },
    topRight:    { x: result.topRightX,    y: result.topRightY },
    bottomRight: { x: result.bottomRightX, y: result.bottomRightY },
    bottomLeft:  { x: result.bottomLeftX,  y: result.bottomLeftY },
  };
}
