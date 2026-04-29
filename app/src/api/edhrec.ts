import type { CachedCard } from '../db/cards';

const BASE = 'https://json.edhrec.com/pages';

export type SynergyMetric = 'synergy' | 'inclusion';

export type SynergyEntry = {
  name: string;
  score: number;
  image_uri: string;
  edhrecUrl: string;
  scryfall_id?: string;
};

export type SynergyResult = {
  metric: SynergyMetric;
  entries: SynergyEntry[];
};

type EdhrecImageField =
  | string
  | { normal?: string; small?: string; large?: string; png?: string };

type EdhrecCardView = {
  id?: string;
  name: string;
  sanitized?: string;
  url?: string;
  synergy?: number;
  num_decks?: number;
  potential_decks?: number;
  image_uris?: EdhrecImageField[];
  images?: EdhrecImageField[];
};

type EdhrecPage = {
  container?: { json_dict?: { cardlists?: Array<{ cardviews?: EdhrecCardView[] }> } };
};

function pickImageUri(cv: EdhrecCardView): string {
  const candidates = [...(cv.image_uris ?? []), ...(cv.images ?? [])];
  for (const c of candidates) {
    if (typeof c === 'string' && c) return c;
    if (c && typeof c === 'object') {
      const uri = c.normal || c.large || c.png || c.small;
      if (uri) return uri;
    }
  }
  return '';
}

// Direct file origin (cards.scryfall.io) is not rate-limited, unlike
// api.scryfall.com/cards/* redirects. URL pattern is documented as stable.
const scryfallImageById = (id: string) =>
  id.length >= 2
    ? `https://cards.scryfall.io/normal/front/${id[0]}/${id[1]}/${id}.jpg`
    : '';

export function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/\s*\/\/\s*/g, '-')
    .replace(/[,'’"()]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/** Legendary creatures + planeswalkers with "can be your commander". Heuristic. */
export function isCommanderEligible(card: CachedCard): boolean {
  const tl = card.type_line.toLowerCase();
  if (tl.includes('legendary') && tl.includes('creature')) return true;
  return /can be your commander/i.test(card.oracle_text);
}

const MAX_ATTEMPTS = 5;
const BASE_DELAY_MS = 750;

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      const err = new Error('Aborted');
      err.name = 'AbortError';
      reject(err);
    });
  });
}

/** Fetches EDHREC JSON with exponential-backoff retries. 404 → null. Final failure throws. */
async function fetchJson(url: string, signal?: AbortSignal): Promise<EdhrecPage | null> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(url, { signal });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`EDHREC ${res.status}`);
      return (await res.json()) as EdhrecPage;
    } catch (err) {
      if (signal?.aborted) throw err;
      lastErr = err;
      if (attempt < MAX_ATTEMPTS - 1) await wait(BASE_DELAY_MS * 2 ** attempt, signal);
    }
  }
  throw lastErr;
}

function scoreFor(cv: EdhrecCardView, metric: SynergyMetric): number | null {
  if (metric === 'synergy') {
    return typeof cv.synergy === 'number' ? Math.round(cv.synergy * 100) : null;
  }
  if (typeof cv.num_decks === 'number' && typeof cv.potential_decks === 'number' && cv.potential_decks > 0) {
    return Math.round((cv.num_decks / cv.potential_decks) * 100);
  }
  return null;
}

function extractEntries(data: EdhrecPage | null, metric: SynergyMetric): SynergyEntry[] {
  if (!data) return [];
  const out: SynergyEntry[] = [];
  const seen = new Set<string>();
  for (const list of data.container?.json_dict?.cardlists ?? []) {
    for (const cv of list.cardviews ?? []) {
      if (!cv.name || seen.has(cv.name)) continue;
      const score = scoreFor(cv, metric);
      if (score === null) continue;
      seen.add(cv.name);
      const image_uri =
        pickImageUri(cv) || (cv.id ? scryfallImageById(cv.id) : '');
      out.push({
        name: cv.name,
        score,
        image_uri,
        edhrecUrl: cv.url ? `https://edhrec.com${cv.url}` : '',
        scryfall_id: cv.id,
      });
    }
  }
  return out;
}

export async function fetchEdhrecSynergies(card: CachedCard, signal?: AbortSignal): Promise<SynergyResult> {
  const commander = isCommanderEligible(card);
  const metric: SynergyMetric = commander ? 'synergy' : 'inclusion';
  const url = `${BASE}/${commander ? 'commanders' : 'cards'}/${slugify(card.name)}.json`;

  const data = await fetchJson(url, signal);
  const entries = extractEntries(data, metric).sort((a, b) => b.score - a.score).slice(0, 30);
  return { metric, entries };
}
