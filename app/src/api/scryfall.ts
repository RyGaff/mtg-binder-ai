import type { CachedCard } from '../db/cards';

const BASE = 'https://api.scryfall.com';
const HEADERS = { 'User-Agent': 'MTGBinderApp/1.0', Accept: 'application/json' };

type ScryfallCard = {
  id: string;
  name: string;
  set: string;
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
  const card = await get<ScryfallCard>(`${BASE}/cards/${setCode.toLowerCase()}/${collectorNumber}`);
  return normalize(card);
}

export async function fetchCardByName(name: string): Promise<CachedCard> {
  const card = await get<ScryfallCard>(`${BASE}/cards/named?fuzzy=${encodeURIComponent(name)}`);
  return normalize(card);
}

export type SearchResult = { data: ScryfallCard[]; has_more: boolean; next_page?: string };

export async function searchScryfall(query: string, page = 1): Promise<CachedCard[]> {
  const result = await get<SearchResult>(
    `${BASE}/cards/search?q=${encodeURIComponent(query)}&page=${page}&order=name`
  );
  return result.data.map(normalize);
}
