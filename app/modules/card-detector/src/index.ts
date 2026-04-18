import { requireNativeModule } from 'expo-modules-core';
import { VisionCameraProxy } from 'react-native-vision-camera';
import type { Frame, FrameProcessorPlugin } from 'react-native-vision-camera';

export type Point = { x: number; y: number };

export type CardCorners = {
  topLeft:      Point;
  topRight:     Point;
  bottomRight:  Point;
  bottomLeft:   Point;
  confidence:   number;        // 0.0–1.0; use CARD_CONFIDENCE_MIN/STABLE thresholds
  rectifiedUri?: string;       // 400×560 perspective-corrected JPEG, file URI
};

export const CARD_CONFIDENCE_MIN    = 0.35;
export const CARD_CONFIDENCE_STABLE = 0.65;

type RawCorners = {
  topLeftX:     number; topLeftY:     number;
  topRightX:    number; topRightY:    number;
  bottomRightX: number; bottomRightY: number;
  bottomLeftX:  number; bottomLeftY:  number;
  confidence:   number;
  rectifiedUri?: string;
};

function parseRaw(raw: RawCorners): CardCorners {
  return {
    topLeft:     { x: raw.topLeftX,     y: raw.topLeftY     },
    topRight:    { x: raw.topRightX,    y: raw.topRightY    },
    bottomRight: { x: raw.bottomRightX, y: raw.bottomRightY },
    bottomLeft:  { x: raw.bottomLeftX,  y: raw.bottomLeftY  },
    confidence:  raw.confidence,
    rectifiedUri: raw.rectifiedUri,
  };
}

export async function detectCardCorners(imageUri: string): Promise<CardCorners | null> {
  const Native = requireNativeModule('CardDetector');
  const raw: RawCorners | null = await Native.detectCardCorners(imageUri);
  if (!raw) return null;
  return parseRaw(raw);
}

export function initCardDetectorPlugin(): FrameProcessorPlugin | null {
  const p = VisionCameraProxy.initFrameProcessorPlugin('detectCardCornersInFrame', {}) ?? null;
  console.log('[CardDetector] plugin:', p == null ? 'NULL - not registered!' : 'OK');
  return p;
}

export function detectCardCornersInFrame(frame: Frame, plugin: FrameProcessorPlugin): CardCorners | null {
  'worklet';
  const result = plugin.call(frame) as Record<string, number | string | undefined> | null;
  if (!result) return null;
  return {
    topLeft:     { x: result.topLeftX     as number, y: result.topLeftY     as number },
    topRight:    { x: result.topRightX    as number, y: result.topRightY    as number },
    bottomRight: { x: result.bottomRightX as number, y: result.bottomRightY as number },
    bottomLeft:  { x: result.bottomLeftX  as number, y: result.bottomLeftY  as number },
    confidence:  result.confidence as number,
    // Frame processor path does not produce rectifiedUri
  };
}
