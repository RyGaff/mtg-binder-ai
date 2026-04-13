export type ParsedCard = { setCode: string; collectorNumber: string };

// Tokens that look like set codes but aren't — language codes, rarity letters,
// common English words, MTG oracle-text words, and copyright noise.
const SKIP_TOKENS = new Set([
  // Language codes
  'EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'JA', 'KO', 'RU', 'ZH', 'PH', 'CS',
  // Single-letter rarity / shorthand
  'R', 'U', 'C', 'M', 'S', 'T', 'L',
  // Common English words that appear in card text / copyright lines
  'THE', 'AND', 'FOR', 'YOU', 'MAY', 'TAP', 'PUT', 'TOP', 'NEW',
  'COPY', 'CAST', 'EACH', 'FROM', 'YOUR', 'BEEN', 'THAT', 'CARD',
  'WITH', 'LESS', 'MANA', 'AURA', 'NON',
  // Copyright / attribution noise
  'MEE', 'LLC', 'INC', 'LTD', 'ALL', 'TM',
]);

/**
 * Parses raw OCR text from the bottom-left corner of an MTG card.
 *
 * Modern card format: "042/350 R IKO EN"  (number, rarity, set code, language)
 * Older card format:  "IKO 042/350"       (set code, number)
 *
 * Finds the collector number and set code independently so both orderings work.
 * Set-code candidates are drawn only from the last 3 lines of the OCR output
 * to avoid matching words from oracle text in the card body.
 */
export function parseSetAndNumber(text: string): ParsedCard | null {
  const upper = text.toUpperCase();

  // Collector number: 1–4 digits, optionally followed by /total.
  // Strip leading zeros so Scryfall accepts the value (e.g. "0322" → "322").
  const numMatch = upper.match(/\b(\d{1,4})(?:\/\d+)?\b/);
  if (!numMatch) return null;
  const collectorNumber = numMatch[1].replace(/^0+(\d)/, '$1');

  // Anchor set-code search to the bottom copyright line where the format is
  // "{number} {rarity} {SET} {LANG}".  Scanning the full OCR text causes
  // oracle-text words like THE, AND, etc. to win the election even with
  // SKIP_TOKENS, so restrict to the last 3 lines.
  const lines = upper.split('\n');
  const bottomLines = lines.slice(-3).join('\n');

  // Set code: starts with a letter, 2–4 alphanumeric chars, not a skip token
  const setCandidates = [...bottomLines.matchAll(/\b([A-Z][A-Z0-9]{1,3})\b/g)]
    .map(m => m[1])
    .filter(s => !SKIP_TOKENS.has(s) && !/^\d+$/.test(s));

  if (setCandidates.length === 0) return null;

  // Prefer 3-letter all-alpha codes (most MTG sets), fall back to first candidate
  const setCode = (setCandidates.find(s => /^[A-Z]{3}$/.test(s)) ?? setCandidates[0]).toLowerCase();

  return { setCode, collectorNumber };
}

/**
 * Runs OCR on a photo URI and returns parsed set/number.
 * Returns null on parse failure so the caller can show a "try again" prompt.
 *
 * NOTE: the URI must be a local file:// path. Callers are responsible for
 * copying ph:// or content:// asset URIs to a cache file before calling this.
 */
export async function scanCardImage(imageUri: string): Promise<ParsedCard | null> {
  // Dynamic require so Jest tests don't need to mock the native module.
  // NativeModules.TextRecognition is null when the native pod was never linked —
  // this produces "Cannot read property 'recognize' of null" at runtime.
  const TextRecognition = require('react-native-text-recognition').default;
  if (!TextRecognition || typeof TextRecognition.recognize !== 'function') {
    throw new Error(
      'OCR module is not available. Rebuild the app with `expo run:ios` to link native dependencies.'
    );
  }
  const lines: string[] = await TextRecognition.recognize(imageUri);
  return parseSetAndNumber(lines.join('\n'));
}
