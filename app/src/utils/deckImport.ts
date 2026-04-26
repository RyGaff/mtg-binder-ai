import { upsertCard, type CachedCard } from '../db/cards';
import { addCardToDeck, ensureDeckArt, type Board } from '../db/decks';
import { fetchCardByName, normalizeScryfallCard } from '../api/scryfall';

export type ParsedLine = { quantity: number; name: string; board: Board };
export type ParsedDeck = { lines: ParsedLine[] };

// Header label → board key. Tolerant of plurals, casing, trailing punctuation.
// Recognizes the common Arena/MTGO/Moxfield/Archidekt section labels.
const BOARD_HEADERS: Record<string, Board> = {
  'commander': 'commander',
  'commanders': 'commander',
  'deck': 'main',
  'main': 'main',
  'mainboard': 'main',
  'main deck': 'main',
  'sideboard': 'side',
  'side': 'side',
  'considering': 'considering',
  'maybeboard': 'considering',
  'maybe': 'considering',
};

// Matches `<qty> <name>` with optional `(SET) <CN>` suffix. We discard the suffix
// since fetchCardByName resolves on name only — set/CN matching is a future polish.
const LINE_RE = /^(\d+)x?\s+(.+?)(?:\s+\([A-Z0-9]+\)\s+\S+)?\s*$/;

export function parseDeckText(text: string): ParsedDeck {
  const lines: ParsedLine[] = [];
  let board: Board = 'main';
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('//') || line.startsWith('#')) continue;
    const headerKey = line.toLowerCase().replace(/[:\-]+$/, '').trim();
    if (BOARD_HEADERS[headerKey]) { board = BOARD_HEADERS[headerKey]; continue; }
    const m = line.match(LINE_RE);
    if (!m) continue;
    const quantity = parseInt(m[1], 10);
    const name = m[2].trim();
    if (quantity > 0 && name) lines.push({ quantity, name, board });
  }
  return { lines };
}

export type ImportProgress = { done: number; total: number; lastName?: string; failed: string[] };
export type ImportProgressFn = (p: ImportProgress) => void;

/**
 * Resolve each parsed line to a Scryfall card and insert into the deck. Sequential
 * to be polite to Scryfall (their guidance is ≤10 req/sec). Fails per-line are
 * collected so the caller can surface them — one bad name doesn't abort the import.
 */
export async function importDeckCards(
  deckId: number,
  lines: ParsedLine[],
  onProgress?: ImportProgressFn,
): Promise<{ failed: string[] }> {
  const failed: string[] = [];
  let done = 0;
  let artSet = false;
  for (const line of lines) {
    try {
      const card = await fetchCardByName(line.name);
      upsertCard(card);
      addCardToDeck({ deck_id: deckId, scryfall_id: card.scryfall_id, quantity: line.quantity, board: line.board });
      // Set deck art from the first commander or main-deck card we successfully resolve.
      if (!artSet && (line.board === 'commander' || line.board === 'main')) {
        await ensureDeckArt(deckId, card.scryfall_id);
        artSet = true;
      }
    } catch {
      failed.push(line.name);
    }
    done++;
    onProgress?.({ done, total: lines.length, lastName: line.name, failed: [...failed] });
  }
  return { failed };
}

export type ResolveResult = {
  resolved: Map<string, CachedCard>; // key = lowercased name
  unresolved: string[]; // names Scryfall didn't recognize (lowercased)
};

const SCRYFALL_BASE = 'https://api.scryfall.com';
const SCRYFALL_HEADERS = {
  'User-Agent': 'MTGBinderApp/1.0',
  Accept: 'application/json',
  'Content-Type': 'application/json',
};
// Scryfall's /cards/collection accepts up to 75 identifiers per request.
const COLLECTION_CHUNK_SIZE = 75;

/**
 * Resolve a batch of card names to CachedCards via Scryfall's bulk
 * /cards/collection endpoint. Dedupes case-insensitively, chunks into ≤75
 * identifiers per request, and runs chunks sequentially to stay under
 * Scryfall's ≤10 req/sec guidance. Returns the resolved cards keyed by
 * lowercased name (both the canonical card name and the original input name
 * when they differ — e.g. when the input was a card-face name) plus the
 * lowercased names Scryfall didn't recognize.
 *
 * Honors the AbortSignal on every fetch; if aborted mid-chunk, the abort
 * error propagates and no partial results are returned.
 */
export async function resolveDeckCards(
  names: string[],
  signal?: AbortSignal,
): Promise<ResolveResult> {
  // Lowercase + dedupe input. Preserve the first original-case spelling for
  // each unique lowercased name so the Scryfall identifier matches what the
  // user typed.
  const inputByLower = new Map<string, string>();
  for (const raw of names) {
    if (!raw) continue;
    const trimmed = raw.trim();
    if (!trimmed) continue;
    const lower = trimmed.toLowerCase();
    if (!inputByLower.has(lower)) inputByLower.set(lower, trimmed);
  }

  if (inputByLower.size === 0) {
    return { resolved: new Map(), unresolved: [] };
  }

  const uniqueOriginals = Array.from(inputByLower.values());
  const resolved = new Map<string, CachedCard>();

  for (let i = 0; i < uniqueOriginals.length; i += COLLECTION_CHUNK_SIZE) {
    if (signal?.aborted) {
      throw signal.reason instanceof Error
        ? signal.reason
        : new DOMException('Aborted', 'AbortError');
    }
    const chunk = uniqueOriginals.slice(i, i + COLLECTION_CHUNK_SIZE);
    const body = JSON.stringify({
      identifiers: chunk.map((name) => ({ name })),
    });
    const res = await fetch(`${SCRYFALL_BASE}/cards/collection`, {
      method: 'POST',
      headers: SCRYFALL_HEADERS,
      body,
      signal,
    });
    if (!res.ok) {
      throw new Error(`Scryfall ${res.status}: /cards/collection`);
    }
    const json = (await res.json()) as {
      data?: Parameters<typeof normalizeScryfallCard>[0][];
      not_found?: { name?: string }[];
    };
    const data = json.data ?? [];
    const cards = data.map(normalizeScryfallCard);
    for (const card of cards) {
      resolved.set(card.name.toLowerCase(), card);
    }
    // /cards/collection preserves request order on `data`, with `not_found`
    // entries omitted from `data`. Walk chunk + data in parallel, skipping
    // not-found inputs, to map the input lowercased name to the canonical
    // card when they differ (e.g. user typed a face name like "Bruna" and
    // Scryfall returns "Bruna, the Fading Light").
    const notFoundLower = new Set(
      (json.not_found ?? [])
        .map((nf) => (nf.name ?? '').toLowerCase())
        .filter(Boolean),
    );
    let dataIdx = 0;
    for (const original of chunk) {
      const lowerInput = original.toLowerCase();
      if (notFoundLower.has(lowerInput)) continue;
      const matched = cards[dataIdx++];
      if (!matched) continue;
      if (matched.name.toLowerCase() !== lowerInput) {
        resolved.set(lowerInput, matched);
      }
    }
  }

  const unresolved: string[] = [];
  for (const lower of inputByLower.keys()) {
    if (!resolved.has(lower)) unresolved.push(lower);
  }

  return { resolved, unresolved };
}
