import * as ImageManipulator from 'expo-image-manipulator';
import { File, Paths } from 'expo-file-system';
import { detectCardCorners, type CardCorners } from '../../modules/card-detector/src';
import { fetchCardBySetNumber, fetchCardByName } from '../api/scryfall';
import { cacheCard, resolveCardById } from '../api/cards';
import { getCardBySetNumber, isCardStale, type CachedCard } from '../db/cards';
import { findCardByImage, ImageMatch } from '../embeddings/imageSearch';

// ── Types ────────────────────────────────────────────────────────────────────

export type ParsedCard = { setCode: string; collectorNumber: string };

export type ScanResult = {
  strategy:  'set_number' | 'name';
  card:      CachedCard;
  corners:   CardCorners;
  imageW:    number;
  imageH:    number;
  ocrText:   string;
  blText:    string;
};

export type ScanProgress =
  | { step: 'corners_detected'; corners: CardCorners; imageW: number; imageH: number }
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
  new File(uri).copy(dest);
  return dest.uri;
}

async function runOcr(uri: string): Promise<string> {
  const TextRecognition = require('react-native-text-recognition').default;
  if (!TextRecognition || typeof TextRecognition.recognize !== 'function') {
    throw new Error('OCR module not available. Run `expo run:ios` to link native dependencies.');
  }
  const lines: string[] = await TextRecognition.recognize(await resolveToFileUri(uri));
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
      if (/^\d+$/.test(token)) collectorNumber = token;
      else if (/^\d+\/\d+$/.test(token)) collectorNumber = token.split('/')[0];
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

// Card edge length in image pixels from normalized corners.
function edgePx(a: { x: number; y: number }, b: { x: number; y: number }, w: number, h: number): number {
  return Math.sqrt(((b.x - a.x) * w) ** 2 + ((b.y - a.y) * h) ** 2);
}

async function cropRegion(
  uri: string,
  originX: number, originY: number, width: number, height: number,
): Promise<string> {
  const r = await ImageManipulator.manipulateAsync(uri, [{ crop: { originX, originY, width, height } }]);
  return r.uri;
}

export async function scanCard(
  uri: string,
  onProgress?: (p: ScanProgress) => void,
  imageSize?: { width: number; height: number },
  precomputedCorners?: CardCorners | null,
): Promise<ScanResult> {
  // Skip re-running OpenCV when the caller already has a detection for this
  // photo (e.g. the image-embedding path calls detectCardCorners first to
  // produce a rectified crop).
  const corners = precomputedCorners ?? await detectCardCorners(uri);
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
    blCropUri = await cropRegion(cropSource, c.x, c.y, c.w, c.h);
  } else {
    const cardWidthPx = edgePx(corners.bottomLeft, corners.bottomRight, imgW, imgH);
    const cardHeightPx = edgePx(corners.topLeft, corners.bottomLeft, imgW, imgH);
    const blOriginX = clamp(Math.floor(corners.bottomLeft.x * imgW), 0, imgW - 1);
    const blOriginY = clamp(Math.floor(corners.bottomLeft.y * imgH - 0.075 * cardHeightPx), 0, imgH - 1);
    const blWidth   = clamp(Math.ceil(0.45 * cardWidthPx),  1, imgW - blOriginX);
    const blHeight  = clamp(Math.ceil(0.075 * cardHeightPx), 1, imgH - blOriginY);
    blCropUri = await cropRegion(uri, blOriginX, blOriginY, blWidth, blHeight);
  }

  const blText = await runOcr(blCropUri);
  onProgress?.({ step: 'bl_ocr_done', blText });
  const parsed = parseSetAndNumber(blText);
  onProgress?.({ step: 'bl_parsed', parsed });

  if (parsed) {
    // DB precheck: a fresh row by (set, number) skips Scryfall entirely.
    const cachedBySet = getCardBySetNumber(parsed.setCode, parsed.collectorNumber);
    if (cachedBySet && !isCardStale(cachedBySet)) {
      cacheCard(cachedBySet); // warm session cache for repeat scans
      return { strategy: 'set_number', card: cachedBySet, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
    }

    let fetched: CachedCard | null = null;
    try {
      onProgress?.({ step: 'fetching', query: `${parsed.setCode.toUpperCase()} #${parsed.collectorNumber}` });
      fetched = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
    } catch {
      // Scryfall 404 or network error — fall through to name strategy
    }
    if (fetched) {
      cacheCard(fetched); // single write-through; no second Scryfall hit
      return { strategy: 'set_number', card: fetched, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
    }
  }

  // ── Strategy 2: name crop (top area) ─────────────────────────────────────

  let nameCropUri: string;
  if (useRectified) {
    const c = CROP_NAME;
    nameCropUri = await cropRegion(cropSource, c.x, c.y, c.w, c.h);
  } else {
    const cardWidthPx = edgePx(corners.bottomLeft, corners.bottomRight, imgW, imgH);
    const cardHeightPx = edgePx(corners.topLeft, corners.bottomLeft, imgW, imgH);
    const tlOriginX = clamp(Math.floor(corners.topLeft.x * imgW), 0, imgW - 1);
    const tlOriginY = clamp(Math.floor(corners.topLeft.y * imgH), 0, imgH - 1);
    const tlWidth   = clamp(Math.ceil(0.65 * cardWidthPx),  1, imgW - tlOriginX);
    const tlHeight  = clamp(Math.ceil(0.12 * cardHeightPx), 1, imgH - tlOriginY);
    nameCropUri = await cropRegion(uri, tlOriginX, tlOriginY, tlWidth, tlHeight);
  }

  const tlText = await runOcr(nameCropUri);
  onProgress?.({ step: 'name_ocr_done', nameText: tlText });

  const nameLine = tlText.split('\n').find(l => l.trim().length > 0 && !/^\d+$/.test(l.trim()));
  if (!nameLine) throw new Error('No text found in name region');

  onProgress?.({ step: 'fetching', query: `name: ${nameLine.trim()}` });
  const card = await fetchCardByName(nameLine.trim());
  cacheCard(card);
  return { strategy: 'name', card, corners, imageW: imgW, imageH: imgH, ocrText: tlText, blText };
}

// ── Public: scanCardByImage ───────────────────────────────────────────────────

/** Kill-switch for the image-embedding identification path. When
 *  false, scan.tsx skips scanCardByImage and falls straight through to
 *  OCR. Set false while encoder / gallery tuning is underway so OCR
 *  alone drives scan outcomes. The detection pipeline (primary →
 *  line-pair → Otsu) still runs — its rectified crop just goes
 *  unused downstream. */
export const EMBEDDING_SCAN_ENABLED = false;

/** Threshold above which we auto-commit the top-1 match. */
export const MATCH_ACCEPT = 0.75;
/** Threshold below which we reject the match and fall back to OCR. */
const MATCH_MIN = 0.55;

export type ImageScanResult = {
  strategy: 'image';
  match:    ImageMatch;
  card:     CachedCard;
};

/**
 * Try image-embedding identification. Caller MUST pass a rectified
 * 400×560 card crop (from detectCardCorners().rectifiedUri) — the
 * encoder was trained on full-card scans and produces garbage on raw
 * photos with hand/table/background. Returns null when:
 *   - The encoder or embeddings are not ready (no artifacts bundled)
 *   - The top-1 match score is below MATCH_MIN (too uncertain)
 *
 * Above MATCH_ACCEPT, the caller should auto-commit.
 */
export async function scanCardByImage(rectifiedUri: string): Promise<ImageScanResult | null> {
  const match = await findCardByImage(rectifiedUri);
  if (!match || match.score < MATCH_MIN) return null;
  const card = await resolveCardById(match.scryfallId);
  return { strategy: 'image', match, card };
}
