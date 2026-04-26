import type { Board } from '../db/decks';
import type { ParsedLine } from '../utils/deckImport';

export type DeckSource = 'moxfield' | 'archidekt';
export type ParsedSource = { source: DeckSource; id: string };
export type FetchedDeck = { name: string; format: string; lines: ParsedLine[] };

const HEADERS = { 'User-Agent': 'MTGBinderApp/1.0', Accept: 'application/json' };

// Allowed format strings (must stay in sync with FORMATS in app/(tabs)/decks.tsx).
const FORMATS = ['Commander', 'Standard', 'Modern', 'Legacy', 'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other'] as const;
type Format = typeof FORMATS[number];

// Lowercased lookup → canonical display value. Anything unmatched falls back to 'Commander'
// (the dominant format on Moxfield/Archidekt).
const FORMAT_LOOKUP: Record<string, Format> = (() => {
  const map: Record<string, Format> = {};
  for (const f of FORMATS) map[f.toLowerCase()] = f;
  return map;
})();

function normalizeFormat(raw: string | null | undefined): Format {
  if (!raw) return 'Commander';
  return FORMAT_LOOKUP[raw.toLowerCase().trim()] ?? 'Commander';
}

// Moxfield: moxfield.com/decks/<id> with optional www., scheme, trailing slash.
const MOXFIELD_RE = /^(?:https?:\/\/)?(?:www\.)?moxfield\.com\/decks\/([A-Za-z0-9_-]+)\/?$/i;
// Archidekt: archidekt.com/decks/<id> with optional trailing slash and optional slug path.
const ARCHIDEKT_RE = /^(?:https?:\/\/)?(?:www\.)?archidekt\.com\/decks\/(\d+)(?:\/[^?#\s]*)?\/?$/i;

export function parseDeckSourceUrl(url: string): ParsedSource | null {
  if (!url) return null;
  const trimmed = url.trim();
  const mox = trimmed.match(MOXFIELD_RE);
  if (mox) return { source: 'moxfield', id: mox[1] };
  const arch = trimmed.match(ARCHIDEKT_RE);
  if (arch) return { source: 'archidekt', id: arch[1] };
  return null;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { headers: HEADERS, signal });
  } catch (err) {
    // Re-throw aborts as-is so callers can detect them; wrap network errors.
    if ((err as { name?: string })?.name === 'AbortError') throw err;
    throw new Error(`Network error fetching ${url}: ${(err as Error).message ?? String(err)}`);
  }
  if (!res.ok) throw new Error(`Fetch failed ${res.status}: ${url}`);
  try {
    return (await res.json()) as T;
  } catch (err) {
    throw new Error(`Invalid JSON from ${url}: ${(err as Error).message ?? String(err)}`);
  }
}

type MoxfieldBoardCard = { quantity: number; card: { name: string } };
type MoxfieldBoard = { cards?: Record<string, MoxfieldBoardCard> };
type MoxfieldDeck = {
  name: string;
  format: string;
  boards?: {
    mainboard?: MoxfieldBoard;
    sideboard?: MoxfieldBoard;
    commanders?: MoxfieldBoard;
    maybeboard?: MoxfieldBoard;
  };
};

const MOXFIELD_BOARD_MAP: Record<string, Board> = {
  mainboard: 'main',
  sideboard: 'side',
  commanders: 'commander',
  maybeboard: 'considering',
};

function flattenMoxfieldBoard(board: MoxfieldBoard | undefined, target: Board): ParsedLine[] {
  if (!board?.cards) return [];
  const out: ParsedLine[] = [];
  for (const key of Object.keys(board.cards)) {
    const entry = board.cards[key];
    const name = entry?.card?.name;
    const quantity = entry?.quantity;
    if (!name || !quantity || quantity <= 0) continue;
    out.push({ quantity, name, board: target });
  }
  return out;
}

export async function fetchMoxfieldDeck(id: string, signal?: AbortSignal): Promise<FetchedDeck> {
  const data = await getJson<MoxfieldDeck>(`https://api.moxfield.com/v3/decks/all/${encodeURIComponent(id)}`, signal);
  const lines: ParsedLine[] = [];
  const boards = data.boards ?? {};
  for (const key of Object.keys(MOXFIELD_BOARD_MAP)) {
    const target = MOXFIELD_BOARD_MAP[key];
    lines.push(...flattenMoxfieldBoard((boards as Record<string, MoxfieldBoard | undefined>)[key], target));
  }
  return {
    name: data.name ?? '',
    format: normalizeFormat(data.format),
    lines,
  };
}

type ArchidektCard = {
  quantity: number;
  categories?: string[];
  card: { oracleCard: { name: string } };
};
type ArchidektDeck = {
  name: string;
  format: string;
  cards: ArchidektCard[];
};

function archidektBoardFor(categories: string[] | undefined): Board {
  if (!categories?.length) return 'main';
  if (categories.includes('Commander')) return 'commander';
  if (categories.includes('Sideboard')) return 'side';
  if (categories.includes('Maybeboard')) return 'considering';
  return 'main';
}

export async function fetchArchidektDeck(id: string, signal?: AbortSignal): Promise<FetchedDeck> {
  const data = await getJson<ArchidektDeck>(`https://archidekt.com/api/decks/${encodeURIComponent(id)}/`, signal);
  const lines: ParsedLine[] = [];
  for (const c of data.cards ?? []) {
    const name = c?.card?.oracleCard?.name;
    const quantity = c?.quantity;
    if (!name || !quantity || quantity <= 0) continue;
    lines.push({ quantity, name, board: archidektBoardFor(c.categories) });
  }
  return {
    name: data.name ?? '',
    format: normalizeFormat(data.format),
    lines,
  };
}

export async function fetchDeckFromUrl(url: string, signal?: AbortSignal): Promise<FetchedDeck> {
  const parsed = parseDeckSourceUrl(url);
  if (!parsed) throw new Error(`Unsupported deck URL: ${url}`);
  if (parsed.source === 'moxfield') return fetchMoxfieldDeck(parsed.id, signal);
  return fetchArchidektDeck(parsed.id, signal);
}
