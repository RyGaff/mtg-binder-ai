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
  ocrText:   string;   // raw text from the region that produced the result
  blText:    string;   // always: raw bottom-left crop OCR (for diagnostics)
};

export type ScanProgress =
  | { step: 'corners_detected'; corners: import('../../modules/card-detector/src').CardCorners; imageW: number; imageH: number }
  | { step: 'bl_ocr_done'; blText: string }
  | { step: 'name_ocr_done'; nameText: string };

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
  source.copy(dest);
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
 *
 * Mirrors the Python cardReader.py logic:
 *   tokens = re.split(r"[^a-zA-Z0-9/]", text)
 *   card_id = first purely numeric token, or left side of "NNN/NNN"
 *   set_id  = first exactly 3-char all-alpha token (not a skip token)
 *
 * e.g. "R 0322\nLTR EN\nArtist" → { setCode: 'ltr', collectorNumber: '0322' }
 */
export function parseSetAndNumber(text: string): ParsedCard | null {
  // Split on anything that isn't alphanumeric or '/' — same as Python
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

  // Fallback: accept alphanumeric set codes (e.g. M21) if no pure-alpha found
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

/**
 * Full scanning pipeline:
 * 1. Detect card corners with OpenCV (native module)
 * 2. Crop bottom-left → OCR → parse set/number → Scryfall (Strategy 1)
 * 3. Fallback: crop name region top-left → OCR → fuzzy name lookup (Strategy 2)
 */
export async function scanCard(
  uri: string,
  onProgress?: (p: ScanProgress) => void,
  imageSize?: { width: number; height: number },
): Promise<ScanResult> {
  const corners = await detectCardCorners(uri);
  if (!corners) throw new Error('No card detected in image');

  let imgW: number;
  let imgH: number;
  if (imageSize) {
    imgW = imageSize.width;
    imgH = imageSize.height;
  } else {
    const info = await ImageManipulator.manipulateAsync(uri, []);
    imgW = info.width;
    imgH = info.height;
  }

  onProgress?.({ step: 'corners_detected', corners, imageW: imgW, imageH: imgH });

  const cardWidthPx = Math.sqrt(
    Math.pow((corners.bottomRight.x - corners.bottomLeft.x) * imgW, 2) +
    Math.pow((corners.bottomRight.y - corners.bottomLeft.y) * imgH, 2)
  );
  const cardHeightPx = Math.sqrt(
    Math.pow((corners.bottomLeft.x - corners.topLeft.x) * imgW, 2) +
    Math.pow((corners.bottomLeft.y - corners.topLeft.y) * imgH, 2)
  );

  // Strategy 1: bottom-left crop (collector number + set code)
  // Extend 6% past the detected corner downward — Vision corners land slightly
  // inside the card frame, which clips the bottom text line on borderless cards.
  const blOriginX = clamp(Math.floor(corners.bottomLeft.x * imgW), 0, imgW - 1);
  const blOriginY = clamp(Math.floor(corners.bottomLeft.y * imgH - 0.075 * cardHeightPx), 0, imgH - 1);
  const blWidth   = clamp(Math.ceil(0.45 * cardWidthPx),  1, imgW - blOriginX);
  const blHeight  = clamp(Math.ceil(0.075 * cardHeightPx), 1, imgH - blOriginY);

  const blCrop = await ImageManipulator.manipulateAsync(uri, [
    { crop: { originX: blOriginX, originY: blOriginY, width: blWidth, height: blHeight } },
  ]);
  const blText = await runOcr(blCrop.uri);
  onProgress?.({ step: 'bl_ocr_done', blText });
  const parsed = parseSetAndNumber(blText);

  if (parsed) {
    try {
      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      return { strategy: 'set_number', card, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
    } catch {
      // Scryfall 404 or network error — fall through to name strategy
    }
  }

  // Strategy 2: name crop (top-left region)
  const tlOriginX = clamp(Math.floor(corners.topLeft.x * imgW), 0, imgW - 1);
  const tlOriginY = clamp(Math.floor(corners.topLeft.y * imgH), 0, imgH - 1);
  const tlWidth   = clamp(Math.ceil(0.65 * cardWidthPx),  1, imgW - tlOriginX);
  const tlHeight  = clamp(Math.ceil(0.12 * cardHeightPx), 1, imgH - tlOriginY);

  const tlCrop = await ImageManipulator.manipulateAsync(uri, [
    { crop: { originX: tlOriginX, originY: tlOriginY, width: tlWidth, height: tlHeight } },
  ]);
  const tlText = await runOcr(tlCrop.uri);
  onProgress?.({ step: 'name_ocr_done', nameText: tlText });

  const nameLine = tlText
    .split('\n')
    .find(l => l.trim().length > 0 && !/^\d+$/.test(l.trim()));

  if (!nameLine) throw new Error('No text found in name region');

  const card = await fetchCardByName(nameLine.trim());
  return { strategy: 'name', card, corners, imageW: imgW, imageH: imgH, ocrText: tlText, blText };
}
