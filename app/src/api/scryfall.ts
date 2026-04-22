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
  card_faces?: { image_uris?: { normal?: string } }[];
  prices?: { usd?: string; usd_foil?: string };
  keywords?: string[];
};

function normalize(card: ScryfallCard): CachedCard {
  const imageUri =
    card.image_uris?.normal ??
    card.card_faces?.[0]?.image_uris?.normal ??
    '';
  return {
    scryfall_id: card.id,
    name: card.name,
    set_code: card.set,
    collector_number: card.collector_number,
    mana_cost: card.mana_cost ?? '',
    type_line: card.type_line ?? '',
    oracle_text: card.oracle_text ?? '',
    color_identity: JSON.stringify(card.color_identity),
    image_uri: imageUri,
    prices: JSON.stringify(card.prices ?? {}),
    keywords: JSON.stringify(card.keywords ?? []),
    cached_at: Date.now(),
  };
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: HEADERS });
  if (!res.ok) throw new Error(`Scryfall ${res.status}: ${url}`);
  return res.json();
}

export async function fetchCardById(id: string): Promise<CachedCard> {
  const card = await get<ScryfallCard>(`${BASE}/cards/${id}`);
  return normalize(card);
}

export async function fetchCardBySetNumber(setCode: string, collectorNumber: string): Promise<CachedCard> {
  // Use search endpoint (mirrors Python: /cards/search?q=set:ltr+cn:0322)
  // More robust than direct /cards/:set/:number — handles leading zeros, case etc.
  const url = `${BASE}/cards/search?q=set%3A${encodeURIComponent(setCode.toLowerCase())}+cn%3A${encodeURIComponent(collectorNumber)}`;
  const result = await get<{ data: ScryfallCard[] }>(url);
  if (!result.data?.length) throw new Error(`Scryfall: no card for ${setCode}/${collectorNumber}`);
  return normalize(result.data[0]);
}

export async function fetchCardByName(name: string): Promise<CachedCard> {
  const attempts: (() => Promise<ScryfallCard>)[] = [
    () => get<ScryfallCard>(`${BASE}/cards/named?exact=${encodeURIComponent(name)}`),
    () => get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(name)}`),
  ];
  const frontFace = name.split(/\s*\/\/\s*/)[0];
  if (frontFace && frontFace !== name) {
    attempts.push(() =>
      get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(frontFace)}`)
    );
  }

  let lastErr: unknown;
  for (const run of attempts) {
    try {
      const card = await run();
      return normalize(card);
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
  prices: {
    usd: string | null;
    usd_foil: string | null;
  };
};

export async function searchScryfall(query: string, page = 1): Promise<CachedCard[]> {
  const result = await get<SearchResult>(
    `${BASE}/cards/search?q=${encodeURIComponent(query)}&page=${page}&order=name`
  );
  return result.data.map(normalize);
}

export async function fetchPrintings(name: string): Promise<PrintingSummary[]> {
  const query = `!"${name}"`;
  const url = `${BASE}/cards/search?q=${encodeURIComponent(query)}&unique=prints&order=released&dir=desc`;
  const result = await get<SearchResult>(url);
  return result.data.map(c => ({
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
