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
  layout?: string;
  all_parts?: { id: string; component: string; name: string; type_line?: string }[];
  prices?: { usd?: string; usd_foil?: string };
  keywords?: string[];
};

// Layouts where each face has its own image URI. Split/flip/adventure share one image.
const TWO_IMAGE_LAYOUTS = new Set([
  'transform', 'modal_dfc', 'double_faced_token', 'art_series', 'reversible_card',
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

function normalize(card: ScryfallCard): CachedCard {
  const imageUri = card.image_uris?.normal ?? card.card_faces?.[0]?.image_uris?.normal ?? '';
  const hasTwoImages = card.layout ? TWO_IMAGE_LAYOUTS.has(card.layout) : false;
  const imageUriBack = hasTwoImages ? (card.card_faces?.[1]?.image_uris?.normal ?? '') : '';
  // Any layout with a card_faces array carries per-face text (transform, mdfc, split,
  // flip, adventure). Meld and leveler have no card_faces; top-level fields stay populated.
  const faces: CardFace[] = (card.card_faces?.length ?? 0) >= 2 ? card.card_faces!.map(normalizeFace) : [];
  const front = faces[0];
  const relatedParts: RelatedPart[] = (card.all_parts ?? [])
    .filter((p): p is RelatedPart => p.component === 'meld_part' || p.component === 'meld_result')
    .map((p) => ({ id: p.id, component: p.component, name: p.name }));
  return {
    scryfall_id: card.id,
    name: card.name,
    set_code: card.set,
    collector_number: card.collector_number,
    mana_cost: card.mana_cost || front?.mana_cost || '',
    type_line: card.type_line || front?.type_line || '',
    oracle_text: card.oracle_text || front?.oracle_text || '',
    color_identity: JSON.stringify(card.color_identity),
    image_uri: imageUri,
    image_uri_back: imageUriBack,
    card_faces: JSON.stringify(faces),
    all_parts: JSON.stringify(relatedParts),
    prices: JSON.stringify(card.prices ?? {}),
    keywords: JSON.stringify(card.keywords ?? []),
    cached_at: Date.now(),
  };
}

// Scryfall asks for ~100ms between requests. Serialize + space outbound calls.
const MIN_GAP_MS = 100;
let requestChain: Promise<unknown> = Promise.resolve();
let lastFinish = 0;

function throttled<T>(run: () => Promise<T>): Promise<T> {
  const next = requestChain.then(async () => {
    const wait = Math.max(0, MIN_GAP_MS - (Date.now() - lastFinish));
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
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
  return throttled(async () => {
    const res = await fetch(url, { headers: HEADERS, signal });
    if (!res.ok) {
      const err = new Error(`Scryfall ${res.status}: ${url}`) as Error & { status?: number };
      err.status = res.status;
      throw err;
    }
    return res.json();
  });
}

export async function fetchCardById(id: string, signal?: AbortSignal): Promise<CachedCard> {
  return normalize(await get<ScryfallCard>(`${BASE}/cards/${id}`, signal));
}

export async function fetchCardBySetNumber(setCode: string, collectorNumber: string, signal?: AbortSignal): Promise<CachedCard> {
  // Use search endpoint (more robust than /cards/:set/:number — handles leading zeros, case).
  const url = `${BASE}/cards/search?q=set%3A${encodeURIComponent(setCode.toLowerCase())}+cn%3A${encodeURIComponent(collectorNumber)}`;
  const result = await get<{ data: ScryfallCard[] }>(url, signal);
  if (!result.data?.length) throw new Error(`Scryfall: no card for ${setCode}/${collectorNumber}`);
  return normalize(result.data[0]);
}

export async function fetchCardByName(name: string, signal?: AbortSignal): Promise<CachedCard> {
  const attempts: (() => Promise<ScryfallCard>)[] = [
    () => get<ScryfallCard>(`${BASE}/cards/named?exact=${encodeURIComponent(name)}`, signal),
    () => get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(name)}`, signal),
  ];
  const front = name.split(/\s*\/\/\s*/)[0];
  if (front && front !== name) {
    attempts.push(() => get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(front)}`, signal));
  }

  let lastErr: unknown;
  for (const run of attempts) {
    try {
      return normalize(await run());
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr ?? new Error(`Could not resolve card: ${name}`);
}

export type SearchResult = { data: ScryfallCard[]; has_more: boolean; next_page?: string };

export type PrintingSummary = {
  scryfall_id: string;
  set_code: string;
  set_name: string;
  collector_number: string;
  image_uri: string;
  prices: { usd: string | null; usd_foil: string | null };
};

export async function searchScryfall(query: string, page = 1, signal?: AbortSignal): Promise<CachedCard[]> {
  const result = await get<SearchResult>(
    `${BASE}/cards/search?q=${encodeURIComponent(query)}&page=${page}&order=name`,
    signal
  );
  return result.data.map(normalize);
}

export async function fetchPrintings(name: string, signal?: AbortSignal): Promise<PrintingSummary[]> {
  const url = `${BASE}/cards/search?q=${encodeURIComponent(`!"${name}"`)}&unique=prints&order=released&dir=desc`;
  const result = await get<SearchResult>(url, signal);
  return result.data.map((c) => ({
    scryfall_id: c.id,
    set_code: c.set,
    set_name: c.set_name ?? '',
    collector_number: c.collector_number,
    image_uri: c.image_uris?.normal ?? c.card_faces?.[0]?.image_uris?.normal ?? '',
    prices: {
      usd: c.prices?.usd ?? null,
      usd_foil: c.prices?.usd_foil ?? null,
    },
  }));
}
