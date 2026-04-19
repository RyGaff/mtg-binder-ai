import * as ImageManipulator from 'expo-image-manipulator';
import { File, Paths } from 'expo-file-system';
import { detectCardCorners } from '../../modules/card-detector/src';
import { fetchCardBySetNumber, fetchCardByName } from '../api/scryfall';
import type { CachedCard } from '../db/cards';

// ── Types ────────────────────────────────────────────────────────────────────

export type ParsedCard = { setCode: string; collectorNumber: string };

export type ScanResult = {
  strategy:  'set_number' | 'name';
  card:      CachedCard;
  corners:   import('../../modules/card-detector/src').CardCorners;
  imageW:    number;
  imageH:    number;
  ocrText:   string;
  blText:    string;
};

export type ScanProgress =
  | { step: 'corners_detected'; corners: import('../../modules/card-detector/src').CardCorners; imageW: number; imageH: number }
  | { step: 'bl_ocr_done';     blText: string }
  | { step: 'bl_parsed';       parsed: ParsedCard | null }
  | { step: 'fetching';        query: string }
  | { step: 'name_ocr_done';   nameText: string };

// Fixed pixel crop regions on the 400×560 rectified card image
const CROP_BOTTOM_LEFT = { x: 8,  y: 504, w: 130, h: 40 };
const CROP_NAME        = { x: 16, y: 22,  w: 310, h: 36 };

// ── Helpers ──────────────────────────────────────────────────────────────────

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
  source.copy(dest);
  return dest.uri;
}

async function runOcr(uri: string): Promise<string> {
  const TextRecognition = require('react-native-text-recognition').default;
  if (!TextRecognition || typeof TextRecognition.recognize !== 'function') {
    throw new Error('OCR module not available. Run `expo run:ios` to link native dependencies.');
  }
  const resolvedUri = await resolveToFileUri(uri);
  const lines: string[] = await TextRecognition.recognize(resolvedUri);
  return lines.join('\n');
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ── Public: parseSetAndNumber ────────────────────────────────────────────────

export function parseSetAndNumber(text: string): ParsedCard | null {
  const tokens = text.toUpperCase().split(/[^A-Z0-9/]+/).filter(Boolean);
  let collectorNumber = '';
  let setCode = '';
  for (const token of tokens) {
    if (!collectorNumber) {
      if (/^\d+$/.test(token)) {
        collectorNumber = token;
      } else if (/^\d+\/\d+$/.test(token)) {
        collectorNumber = token.split('/')[0];
      }
    }
    if (!setCode && token.length === 3 && /^[A-Z]+$/.test(token) && !SKIP_TOKENS.has(token)) {
      setCode = token;
    }
    if (collectorNumber && setCode) break;
  }
  if (collectorNumber && !setCode) {
    for (const token of tokens) {
      if (/^[A-Z][A-Z0-9]{2}$/.test(token) && !SKIP_TOKENS.has(token)) {
        setCode = token;
        break;
      }
    }
  }
  if (!collectorNumber || !setCode) return null;
  return { setCode: setCode.toLowerCase(), collectorNumber };
}

// ── Public: scanCard ─────────────────────────────────────────────────────────

export async function scanCard(
  uri: string,
  onProgress?: (p: ScanProgress) => void,
  imageSize?: { width: number; height: number },
): Promise<ScanResult> {
  const corners = await detectCardCorners(uri);
  if (!corners) throw new Error('No card detected in image');

  const useRectified = !!corners.rectifiedUri;
  const cropSource = corners.rectifiedUri ?? uri;

  let imgW: number;
  let imgH: number;
  if (useRectified) {
    // Rectified image is always 400×560 by construction
    imgW = 400;
    imgH = 560;
  } else if (imageSize) {
    imgW = imageSize.width;
    imgH = imageSize.height;
  } else {
    const info = await ImageManipulator.manipulateAsync(uri, []);
    imgW = info.width;
    imgH = info.height;
  }

  onProgress?.({ step: 'corners_detected', corners, imageW: imgW, imageH: imgH });

  // ── Strategy 1: bottom-left (set/collector number) ──────────────────────────

  let blCropUri: string;
  if (useRectified) {
    const c = CROP_BOTTOM_LEFT;
    const result = await ImageManipulator.manipulateAsync(cropSource, [
      { crop: { originX: c.x, originY: c.y, width: c.w, height: c.h } },
    ]);
    blCropUri = result.uri;
  } else {
    const cardWidthPx = Math.sqrt(
      Math.pow((corners.bottomRight.x - corners.bottomLeft.x) * imgW, 2) +
      Math.pow((corners.bottomRight.y - corners.bottomLeft.y) * imgH, 2)
    );
    const cardHeightPx = Math.sqrt(
      Math.pow((corners.bottomLeft.x - corners.topLeft.x) * imgW, 2) +
      Math.pow((corners.bottomLeft.y - corners.topLeft.y) * imgH, 2)
    );
    const blOriginX = clamp(Math.floor(corners.bottomLeft.x * imgW), 0, imgW - 1);
    const blOriginY = clamp(Math.floor(corners.bottomLeft.y * imgH - 0.075 * cardHeightPx), 0, imgH - 1);
    const blWidth   = clamp(Math.ceil(0.45 * cardWidthPx),  1, imgW - blOriginX);
    const blHeight  = clamp(Math.ceil(0.075 * cardHeightPx), 1, imgH - blOriginY);
    const blCrop = await ImageManipulator.manipulateAsync(uri, [
      { crop: { originX: blOriginX, originY: blOriginY, width: blWidth, height: blHeight } },
    ]);
    blCropUri = blCrop.uri;
  }

  const blText = await runOcr(blCropUri);
  onProgress?.({ step: 'bl_ocr_done', blText });
  const parsed = parseSetAndNumber(blText);
  onProgress?.({ step: 'bl_parsed', parsed });

  if (parsed) {
    try {
      onProgress?.({ step: 'fetching', query: `${parsed.setCode.toUpperCase()} #${parsed.collectorNumber}` });
      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      return { strategy: 'set_number', card, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
    } catch {
      // Scryfall 404 or network error — fall through to name strategy
    }
  }

  // ── Strategy 2: name crop (top area) ─────────────────────────────────────

  let nameCropUri: string;
  if (useRectified) {
    const c = CROP_NAME;
    const result = await ImageManipulator.manipulateAsync(cropSource, [
      { crop: { originX: c.x, originY: c.y, width: c.w, height: c.h } },
    ]);
    nameCropUri = result.uri;
  } else {
    const cardWidthPx = Math.sqrt(
      Math.pow((corners.bottomRight.x - corners.bottomLeft.x) * imgW, 2) +
      Math.pow((corners.bottomRight.y - corners.bottomLeft.y) * imgH, 2)
    );
    const cardHeightPx = Math.sqrt(
      Math.pow((corners.bottomLeft.x - corners.topLeft.x) * imgW, 2) +
      Math.pow((corners.bottomLeft.y - corners.topLeft.y) * imgH, 2)
    );
    const tlOriginX = clamp(Math.floor(corners.topLeft.x * imgW), 0, imgW - 1);
    const tlOriginY = clamp(Math.floor(corners.topLeft.y * imgH), 0, imgH - 1);
    const tlWidth   = clamp(Math.ceil(0.65 * cardWidthPx),  1, imgW - tlOriginX);
    const tlHeight  = clamp(Math.ceil(0.12 * cardHeightPx), 1, imgH - tlOriginY);
    const tlCrop = await ImageManipulator.manipulateAsync(uri, [
      { crop: { originX: tlOriginX, originY: tlOriginY, width: tlWidth, height: tlHeight } },
    ]);
    nameCropUri = tlCrop.uri;
  }

  const tlText = await runOcr(nameCropUri);
  onProgress?.({ step: 'name_ocr_done', nameText: tlText });

  const nameLine = tlText
    .split('\n')
    .find(l => l.trim().length > 0 && !/^\d+$/.test(l.trim()));

  if (!nameLine) throw new Error('No text found in name region');

  onProgress?.({ step: 'fetching', query: `name: ${nameLine.trim()}` });
  const card = await fetchCardByName(nameLine.trim());
  return { strategy: 'name', card, corners, imageW: imgW, imageH: imgH, ocrText: tlText, blText };
}
