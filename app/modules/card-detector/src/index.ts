import { requireNativeModule } from 'expo-modules-core';
import { VisionCameraProxy } from 'react-native-vision-camera';
import type { Frame, FrameProcessorPlugin } from 'react-native-vision-camera';

export type Point = { x: number; y: number };

/** Which detection path produced the corners.
 *  - primary:    Canny + contour quad (standard scan path)
 *  - lineinterp: Hough-cluster rectangle (rescue for occluded edges)
 *  - otsu:       Otsu threshold segmentation (rescue for weak-gradient edges)
 * Native already applies source-aware expansion before writing the
 * rectified crop, so consumers normally don't need to branch on this —
 * exposed for diagnostics and confidence-weighting. */
export type CardSource = 'primary' | 'lineinterp' | 'otsu';

export type CardCorners = {
  topLeft:      Point;
  topRight:     Point;
  bottomRight:  Point;
  bottomLeft:   Point;
  confidence:   number;        // 0.0–1.0; use CARD_CONFIDENCE_MIN/STABLE thresholds
  source:       CardSource;
  rectifiedUri?: string;       // 400×560 perspective-corrected JPEG, file URI
};

export const CARD_CONFIDENCE_MIN    = 0.35;
export const CARD_CONFIDENCE_STABLE = 0.45;

type RawCorners = {
  topLeftX:     number; topLeftY:     number;
  topRightX:    number; topRightY:    number;
  bottomRightX: number; bottomRightY: number;
  bottomLeftX:  number; bottomLeftY:  number;
  confidence:   number;
  source?:      string;
  rectifiedUri?: string;
};

function toCardSource(raw?: string): CardSource {
  'worklet';
  return raw === 'lineinterp' || raw === 'otsu' ? raw : 'primary';
}

function parseRaw(raw: RawCorners): CardCorners {
  return {
    topLeft:     { x: raw.topLeftX,     y: raw.topLeftY     },
    topRight:    { x: raw.topRightX,    y: raw.topRightY    },
    bottomRight: { x: raw.bottomRightX, y: raw.bottomRightY },
    bottomLeft:  { x: raw.bottomLeftX,  y: raw.bottomLeftY  },
    confidence:  raw.confidence,
    source:      toCardSource(raw.source),
    rectifiedUri: raw.rectifiedUri,
  };
}

export async function detectCardCorners(imageUri: string): Promise<CardCorners | null> {
  const Native = requireNativeModule('CardDetector');
  const raw: RawCorners & {
    _error?: string;
    resolvedPath?: string;
    fileExists?: boolean;
    stats?: Record<string, number>;
  } | null = await Native.detectCardCorners(imageUri);
  if (!raw) return null;
  if (raw._error) {
    const s = raw.stats ?? {};
    const statStr = raw.stats
      ? ` [cont=${s.contoursTotal} 4v=${s.passed4Vertex} area=${s.passedMinArea} conv=${s.passedConvex} ang=${s.passedAngles} AR=${s.passedAR}]`
      : '';
    const pathStr = raw.resolvedPath
      ? ` path=${raw.resolvedPath} exists=${raw.fileExists}`
      : '';
    throw new Error(`Photo detection failed: ${raw._error}${statStr}${pathStr}`);
  }
  return parseRaw(raw);
}

export function initCardDetectorPlugin(): FrameProcessorPlugin | null {
  return VisionCameraProxy.initFrameProcessorPlugin('detectCardCornersInFrame', {}) ?? null;
}

export type DetectionStats = {
  medianLuma:    number;
  edgePixels:    number;
  contoursTotal: number;
  passed4Vertex: number;
  passedMinArea: number;
  passedConvex:  number;
  passedAngles:  number;
  passedAR:      number;
};

export type FrameDebug = {
  pixelFormat: string | null;
  frameW:      number;
  frameH:      number;
  stats:       DetectionStats | null;
};

export function detectCardCornersInFrame(
  frame: Frame,
  plugin: FrameProcessorPlugin,
): { corners: CardCorners | null; debug: FrameDebug } {
  'worklet';
  const result = plugin.call(frame) as Record<string, unknown> | null;
  if (!result) {
    return { corners: null, debug: { pixelFormat: null, frameW: 0, frameH: 0, stats: null } };
  }
  const rawStats = result.stats as Record<string, number> | undefined;
  const stats: DetectionStats | null = rawStats
    ? {
        medianLuma:    rawStats.medianLuma    ?? 0,
        edgePixels:    rawStats.edgePixels    ?? 0,
        contoursTotal: rawStats.contoursTotal ?? 0,
        passed4Vertex: rawStats.passed4Vertex ?? 0,
        passedMinArea: rawStats.passedMinArea ?? 0,
        passedConvex:  rawStats.passedConvex  ?? 0,
        passedAngles:  rawStats.passedAngles  ?? 0,
        passedAR:      rawStats.passedAR      ?? 0,
      }
    : null;
  const debug: FrameDebug = {
    pixelFormat: (result.pixelFormat as string) ?? null,
    frameW:      (result.frameW as number) ?? 0,
    frameH:      (result.frameH as number) ?? 0,
    stats,
  };
  if (result._debug) {
    return { corners: null, debug };
  }
  return {
    corners: {
      topLeft:     { x: result.topLeftX     as number, y: result.topLeftY     as number },
      topRight:    { x: result.topRightX    as number, y: result.topRightY    as number },
      bottomRight: { x: result.bottomRightX as number, y: result.bottomRightY as number },
      bottomLeft:  { x: result.bottomLeftX  as number, y: result.bottomLeftY  as number },
      confidence:  result.confidence as number,
      source:      toCardSource(result.source as string | undefined),
    },
    debug,
  };
}

/**
 * Run the bundled image encoder on the file at `imageUri`. Returns a
 * 256-length Float32Array (L2-normalized by the model) or null when the
 * encoder asset is not bundled in this build, the image can't be decoded,
 * or the native call fails.
 */
export async function encodeImage(imageUri: string): Promise<Float32Array | null> {
  const Native = requireNativeModule('CardDetector');
  // iOS returns { embedding: number[] } | { error: string }
  // Android returns number[] | null
  const raw: unknown = await Native.encodeImage(imageUri);
  if (raw == null) return null;
  if (Array.isArray(raw)) return new Float32Array(raw);
  if (typeof raw === 'object') {
    const obj = raw as { embedding?: number[]; error?: string };
    if (obj.error) {
      console.warn('[CardDetector.encodeImage]', obj.error);
      return null;
    }
    if (obj.embedding) return new Float32Array(obj.embedding);
  }
  return null;
}
