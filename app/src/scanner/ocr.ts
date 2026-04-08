// SET_CODE: 2–4 alphanumeric chars. COLLECTOR_NUMBER: digits (optionally /total).
const SET_NUMBER_RE = /\b([a-z0-9]{2,4})\s+(\d{1,4})(?:\/\d+)?(?:\b|$)/i;

export type ParsedCard = { setCode: string; collectorNumber: string };

/**
 * Parses raw OCR text from the bottom-left corner of an MTG card.
 * Returns set code + collector number, or null if not found.
 */
export function parseSetAndNumber(text: string): ParsedCard | null {
  const match = text.match(SET_NUMBER_RE);
  if (!match) return null;
  return { setCode: match[1].toLowerCase(), collectorNumber: match[2] };
}

/**
 * Runs ML Kit OCR on a photo URI and returns parsed set/number.
 * Returns null on parse failure so the caller can show a "try again" prompt.
 */
export async function scanCardImage(imageUri: string): Promise<ParsedCard | null> {
  // Dynamic require so Jest tests don't need to mock the native module
  const TextRecognition = require('@react-native-ml-kit/text-recognition').default;
  const result = await TextRecognition.recognize(imageUri);
  return parseSetAndNumber(result.text);
}
