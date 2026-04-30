import type { CachedCard } from '../db/cards';

const BASE = 'https://api.scryfall.com';
const HEADERS = { 'User-Agent': 'MTGBinderApp/1.0', Accept: 'application/json' };

type ScryfallCard = {
  id: string;
  name: string;
  set: string;
  set_name?: string;
  collector_number: string;
  mana_cost?: string;
  type_line?: string;
  oracle_text?: string;
  color_identity: string[];
  image_uris?: { normal?: string; large?: string };
  card_faces?: {
    name?: string;
    mana_cost?: string;
    type_line?: string;
    oracle_text?: string;
    image_uris?: { normal?: string };
  }[];
  layout?: string; // Meld, flip, split, normal, transform
  all_parts?: { id: string; component: string; name: string; type_line?: string }[];
  prices?: { usd?: string; usd_foil?: string };
  keywords?: string[];
};

// Layouts where each face has its own image URI. Split/flip/adventure share one image.
const TWO_IMAGE_LAYOUTS = new Set([
  'transform', 'modal_dfc', 'double_faced_token', 'art_series', 'reversible_card', 'meld',
]);

export type RelatedPart = {
  id: string;
  component: 'meld_part' | 'meld_result';
  name: string;
};

export type CardFace = {
  name: string;
  mana_cost: string;
  type_line: string;
  oracle_text: string;
  image_uri: string;
};

function normalizeFace(f: NonNullable<ScryfallCard['card_faces']>[number]): CardFace {
  return {
    name: f.name ?? '',
    mana_cost: f.mana_cost ?? '',
    type_line: f.type_line ?? '',
    oracle_text: f.oracle_text ?? '',
    image_uri: f.image_uris?.normal ?? '',
  };
}

export function normalizeScryfallCard(card: ScryfallCard): CachedCard {
  const imageUri = card.image_uris?.normal ?? card.card_faces?.[0]?.image_uris?.normal ?? '';
  const hasTwoImages = card.layout ? TWO_IMAGE_LAYOUTS.has(card.layout) : false;
  const imageUriBack = hasTwoImages ? (card.card_faces?.[1]?.image_uris?.normal ?? '') : '';
  // Any layout with a card_faces array carries per-face text (transform, mdfc, split,
  // flip, adventure). Meld and leveler have no card_faces; top-level fields stay populated.
  const faces: CardFace[] = (card.card_faces?.length ?? 0) >= 2 ? card.card_faces!.map(normalizeFace) : [];
  const front = faces[0];
  const back = faces[1];
  // Cards with two castable faces (split, adventure, modal_dfc) carry a mana
  // cost on each face. Join with " // " so renderers can split them. Scryfall
  // already does this for `split` at the top level; we mirror it for the rest.
  const combinedCost =
    front?.mana_cost && back?.mana_cost
      ? `${front.mana_cost} // ${back.mana_cost}`
      : (card.mana_cost || front?.mana_cost || '');
  const relatedParts: RelatedPart[] = (card.all_parts ?? [])
    .filter((p): p is RelatedPart => p.component === 'meld_part' || p.component === 'meld_result')
    .map((p) => ({ id: p.id, component: p.component, name: p.name }));
  return {
    scryfall_id: card.id,
    name: card.name,
    set_code: card.set,
    collector_number: card.collector_number,
    mana_cost: combinedCost,
    type_line: card.type_line || front?.type_line || '',
    oracle_text: card.oracle_text || front?.oracle_text || '',
    color_identity: JSON.stringify(card.color_identity),
    image_uri: imageUri,
    image_uri_back: imageUriBack,
    card_faces: JSON.stringify(faces),
    all_parts: JSON.stringify(relatedParts),
    prices: JSON.stringify(card.prices ?? {}),
    keywords: JSON.stringify(card.keywords ?? []),
    layout: card.layout ?? 'normal',
    cached_at: Date.now(),
  };
}

// Scryfall hard limits: /cards/search, /cards/named, /cards/collection,
// /cards/random are 2/sec (500ms). All other endpoints are 10/sec (100ms).
// One global chain so concurrent callers serialize; gap picked per URL.
const SLOW_GAP_MS = 500;
const FAST_GAP_MS = 100;
const SLOW_PATHS = ['/cards/search', '/cards/named', '/cards/collection', '/cards/random'];

let requestChain: Promise<unknown> = Promise.resolve();
let lastFinish = 0;

function gapFor(url: string): number {
  return SLOW_PATHS.some((p) => url.includes(p)) ? SLOW_GAP_MS : FAST_GAP_MS;
}

// Hermes (RN) doesn't ship DOMException, so synthesize an AbortError-shaped
// Error: name='AbortError' is what fetch/AbortController consumers check.
function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const e = new Error('Aborted');
  e.name = 'AbortError';
  return e;
}

/** Serialize + space any Scryfall call. Exposed so non-GET callers (e.g. POST
 * /cards/collection in deckImport) share the same outbound queue. Honors the
 * caller's AbortSignal during the throttle wait so aborts don't keep the
 * chain pinned on a hung in-flight fetch. */
export function throttledScryfall<T>(
  url: string,
  run: () => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  const next = requestChain.then(async () => {
    if (signal?.aborted) throw abortError(signal);
    const waitMs = Math.max(0, gapFor(url) - (Date.now() - lastFinish));
    if (waitMs > 0) {
      await new Promise<void>((resolve, reject) => {
        const t = setTimeout(resolve, waitMs);
        signal?.addEventListener(
          'abort',
          () => {
            clearTimeout(t);
            reject(abortError(signal));
          },
          { once: true },
        );
      });
    }
    if (signal?.aborted) throw abortError(signal);
    try {
      return await run();
    } finally {
      lastFinish = Date.now();
    }
  });
  requestChain = next.catch(() => {});
  return next;
}

async function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  return throttledScryfall(
    url,
    async () => {
      const res = await fetch(url, { headers: HEADERS, signal });
      if (!res.ok) {
        const err = new Error(`Scryfall ${res.status}: ${url}`) as Error & { status?: number };
        err.status = res.status;
        throw err;
      }
      return res.json();
    },
    signal,
  );
}

export async function fetchCardById(id: string, signal?: AbortSignal): Promise<CachedCard> {
  return normalizeScryfallCard(await get<ScryfallCard>(`${BASE}/cards/${id}`, signal));
}

/** Fetch just the art_crop URI for a card. Falls back to face[0] for split/dfc layouts. */
export async function fetchArtCrop(id: string, signal?: AbortSignal): Promise<string | null> {
  const card = await get<{
    image_uris?: { art_crop?: string };
    card_faces?: Array<{ image_uris?: { art_crop?: string } }>;
  }>(`${BASE}/cards/${id}`, signal);
  return card.image_uris?.art_crop ?? card.card_faces?.[0]?.image_uris?.art_crop ?? null;
}

export async function fetchCardBySetNumber(setCode: string, collectorNumber: string, signal?: AbortSignal): Promise<CachedCard> {
  // Use search endpoint (more robust than /cards/:set/:number — handles leading zeros, case).
  const url = `${BASE}/cards/search?q=set%3A${encodeURIComponent(setCode.toLowerCase())}+cn%3A${encodeURIComponent(collectorNumber)}`;
  const result = await get<{ data: ScryfallCard[] }>(url, signal);
  if (!result.data?.length) throw new Error(`Scryfall: no card for ${setCode}/${collectorNumber}`);
  return normalizeScryfallCard(result.data[0]);
}

export async function fetchCardByName(name: string, signal?: AbortSignal): Promise<CachedCard> {
  // Always query the front face only. Multi-face names ("A // B") are split
  // on `//`; single-face names pass through unchanged. Fuzzy match is forgiving
  // of casing/punctuation drift in user-pasted decklists.
  const front = name.split(/\s*\/\/\s*/)[0].trim();
  const card = await get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(front)}`, signal);
  return normalizeScryfallCard(card);
}

export type SearchResult = { data: ScryfallCard[]; has_more: boolean; next_page?: string };

export type PrintingSummary = {
  scryfall_id: string;
  set_code: string;
  set_name: string;
  collector_number: string;
  image_uri: string;
  image_uri_back: string;
  layout: string;
  card_faces: string; // JSON string, matches CachedCard.card_faces
  prices: { usd: string | null; usd_foil: string | null };
};

export async function searchScryfall(query: string, page = 1, signal?: AbortSignal): Promise<CachedCard[]> {
  const result = await get<SearchResult>(
    `${BASE}/cards/search?q=${encodeURIComponent(query)}&page=${page}&order=name`,
    signal
  );
  return result.data.map(normalizeScryfallCard);
}

export async function fetchPrintings(name: string, signal?: AbortSignal): Promise<PrintingSummary[]> {
  const url = `${BASE}/cards/search?q=${encodeURIComponent(`!"${name}"`)}&unique=prints&order=released&dir=desc`;
  const result = await get<SearchResult>(url, signal);
  return result.data.map((c) => {
    const n = normalizeScryfallCard(c);
    return {
      scryfall_id: c.id,
      set_code: c.set,
      set_name: c.set_name ?? '',
      collector_number: c.collector_number,
      image_uri: n.image_uri,
      image_uri_back: n.image_uri_back,
      layout: n.layout,
      card_faces: n.card_faces,
      prices: {
        usd: c.prices?.usd ?? null,
        usd_foil: c.prices?.usd_foil ?? null,
      },
    };
  });
}
