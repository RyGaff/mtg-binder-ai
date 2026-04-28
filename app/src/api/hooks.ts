import { useMemo } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { fetchCardById, searchScryfall, fetchPrintings } from './scryfall';
import {
  getCardById,
  getCardsByIds,
  upsertCardsIfNewer,
  isCardStale,
  searchCardsLocal,
  STALE_MS,
} from '../db/cards';
import type { CachedCard } from '../db/cards';
import { cacheCard, cacheCards, peekSessionCard, setSessionCard, touchSessionCard } from './cards';
import { getPrintingsByName, isPrintingsStale, upsertPrintings } from '../db/printings';
import { getEmbeddingMap } from '../embeddings/parser';
import { similaritySearch } from '../embeddings/similarity';
import { useStore } from '../store/useStore';
import { fetchEdhrecSynergies, type SynergyResult } from './edhrec';

const DAY_MS = STALE_MS;
const WEEK_MS = 7 * DAY_MS;

/** Returns a card, using SQLite cache and falling back to Scryfall.
 *  On network / rate-limit failure, falls back to stale cache. */
export function useCard(scryfallId: string) {
  // Resolve once per id change. Session LRU first (zero IO), then SQLite.
  const seed = useMemo(() => {
    if (!scryfallId) return undefined;
    const fromSession = peekSessionCard(scryfallId);
    if (fromSession && !isCardStale(fromSession)) {
      touchSessionCard(scryfallId); // promote on use
      return fromSession;
    }
    const c = getCardById(scryfallId);
    if (c && !isCardStale(c)) {
      setSessionCard(c); // warm LRU only — row already lives on disk
      return c;
    }
    return undefined;
  }, [scryfallId]);

  return useQuery({
    queryKey: ['card', scryfallId],
    queryFn: async ({ signal }) => {
      const stale = getCardById(scryfallId);
      try {
        const fresh = await fetchCardById(scryfallId, signal);
        cacheCard(fresh); // SQLite + LRU write-through
        return fresh;
      } catch (err) {
        if (stale) return stale; // offline / rate-limit fallback
        throw err;
      }
    },
    enabled: !!scryfallId,
    staleTime: DAY_MS,
    // gcTime tightened: LRU now holds card payloads in memory across mounts,
    // so RQ doesn't need to. Releases image-heavy data when screens unmount.
    gcTime: 60 * 1000,
    initialData: seed,
    initialDataUpdatedAt: seed?.cached_at,
  });
}

/** Search Scryfall and cache results. Falls back to local FTS when offline. */
export function useScryfallSearch(query: string) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: async ({ signal }) => {
      try {
        const results = await searchScryfall(query, 1, signal);
        // Seed-only: search payloads omit fields like `all_parts` that detail
        // payloads carry, so don't clobber existing detailed rows.
        upsertCardsIfNewer(results);
        return results;
      } catch (err) {
        // Don't poison the cache on cancel — RQ aborts in-flight searches as
        // the user types; treating that as "network failed" caches local FTS
        // results for the aborted key.
        if ((err as Error)?.name === 'AbortError' || signal?.aborted) throw err;
        return searchCardsLocal(query);
      }
    },
    enabled: query.length > 1,
    staleTime: 30 * 1000,
    // Image-heavy and ephemeral — release shortly after the search screen
    // unmounts so we don't hold hundreds of card image URIs in memory.
    // `placeholderData: keepPreviousData` only smooths the in-screen
    // query-string transition; it doesn't depend on long gcTime.
    gcTime: 30 * 1000,
    placeholderData: keepPreviousData,
  });
}

/** Pull synergy partners from EDHREC (co-occurrence data across public deck lists). */
export function useSynergyFromCard(seed: CachedCard | null) {
  return useQuery<SynergyResult>({
    queryKey: ['synergy', seed?.scryfall_id ?? ''],
    queryFn: ({ signal }) =>
      seed ? fetchEdhrecSynergies(seed, signal) : Promise.resolve({ metric: 'synergy', entries: [] }),
    enabled: !!seed,
    staleTime: DAY_MS,
    // SQLite is the durable layer; RQ is a thin observer cache. WEEK_MS
    // pinned synergy response arrays in JS heap forever.
    gcTime: 5 * 60 * 1000,
    retry: (count) => count < 5,
    retryDelay: (count) => Math.min(1000 * 2 ** count, 30000),
  });
}

// ---------------------------------------------------------------------------
// Effect category extraction
// ---------------------------------------------------------------------------

const EFFECT_CATEGORIES: { name: string; patterns: RegExp[] }[] = [
  { name: 'damage',   patterns: [/\bdeals?\b/i, /\bdamage\b/i] },
  { name: 'draw',     patterns: [/\bdraw\b.*\bcards?\b/i] },
  { name: 'destroy',  patterns: [/destroy (?:target|all)/i] },
  { name: 'exile',    patterns: [/exile (?:target|all)/i] },
  { name: 'counter',  patterns: [/counter target/i] },
  { name: 'tokens',   patterns: [/\btoken\b/i] },
  { name: 'ramp',     patterns: [/add \{|search your library for a (?:basic )?land/i] },
  { name: 'lifegain', patterns: [/(?:you )?gain (?:\d+ )?life/i] },
  { name: 'discard',  patterns: [/discards? a? cards?|discard (?:their|your) hand/i] },
  { name: 'bounce',   patterns: [/return target/i] },
  { name: 'tutor',    patterns: [/search your library for a card/i] },
  { name: 'pump',     patterns: [/gets \+/i] },
];

const ORACLE_CLAUSE: Record<string, string> = {
  damage:   'o:"deals" o:"damage"',
  draw:     'o:"draw"',
  destroy:  'o:"destroy"',
  exile:    'o:"exile target"',
  counter:  'o:"counter target"',
  tokens:   'o:"token"',
  ramp:     'o:"add {"',
  lifegain: 'o:"gain" o:"life"',
  discard:  'o:"discard"',
  bounce:   'o:"return target"',
  tutor:    'o:"search your library"',
  pump:     'o:"gets +"',
};

export function extractEffectCategories(oracleText: string): string[] {
  return EFFECT_CATEGORIES.filter(cat => cat.patterns.every(p => p.test(oracleText))).map(c => c.name);
}

export function parseManaValue(manaCost: string): number {
  const tokens = manaCost.match(/\{([^}]+)\}/g) ?? [];
  return tokens.reduce((sum, token) => {
    const inner = token.slice(1, -1);
    const n = parseInt(inner, 10);
    if (!isNaN(n)) return sum + n;
    return inner === 'X' ? sum : sum + 1;
  }, 0);
}

export function buildSimilarQuery(card: CachedCard): string {
  const categories = extractEffectCategories(card.oracle_text);
  const mv = parseManaValue(card.mana_cost);
  const colors = JSON.parse(card.color_identity) as string[];
  const parts: string[] = [];
  if (categories.length > 0) {
    const clauses = categories.map(cat => ORACLE_CLAUSE[cat]).filter(Boolean);
    if (clauses.length === 1) parts.push(clauses[0]);
    else parts.push(`(${clauses.join(') or (')})`);
  }
  parts.push(`mv>=${Math.max(0, mv - 1)} mv<=${mv + 1}`);
  if (colors.length > 0) parts.push(`c:${colors.join('')}`);
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Embedding-based similar card search
// ---------------------------------------------------------------------------

/** Fetch top 20 most similar cards via on-device embedding cosine similarity. */
export function useSimilarSearch(card: CachedCard) {
  const embeddingStatus = useStore((s) => s.embeddingStatus);

  return useQuery({
    queryKey: ['similar-embedding', card.scryfall_id],
    queryFn: async ({ signal }) => {
      const index = await getEmbeddingMap();
      // Embedding built from oracle-cards (one printing per unique card); fall back to name lookup.
      const embeddingId = index.idIndex.has(card.scryfall_id)
        ? card.scryfall_id
        : index.byName.get(card.name);
      const results = embeddingId ? similaritySearch(embeddingId, index, 20) : [];

      const ids = results.map((r) => r.scryfallId);
      // freshOnly: stale rows (>24h) drop out and get refetched.
      const cachedMap = getCardsByIds(ids, { freshOnly: true });
      const missing = ids.filter((id) => !cachedMap.has(id));
      const fetched = await Promise.allSettled(missing.map((id) => fetchCardById(id, signal)));
      const fetchedCards = fetched.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []));
      if (fetchedCards.length) cacheCards(fetchedCards); // SQLite + LRU
      for (const c of fetchedCards) cachedMap.set(c.scryfall_id, c);

      return ids.map((id) => cachedMap.get(id)).filter((c): c is CachedCard => !!c);
    },
    enabled: embeddingStatus === 'idle' && !!card?.scryfall_id,
    staleTime: DAY_MS,
    // Same reasoning as useSynergyFromCard: keep RQ memory tight, lean on SQLite.
    gcTime: 5 * 60 * 1000,
    retry: false,
  });
}

// ---------------------------------------------------------------------------
// Printings
// ---------------------------------------------------------------------------

export function buildPrintingsQueryKey(card: CachedCard): [string, string] {
  return ['printings', card.name];
}

/** Fetch all printings of a card sorted newest first. SQLite-backed. */
export function usePrintings(card: CachedCard) {
  return useQuery({
    queryKey: buildPrintingsQueryKey(card),
    queryFn: async ({ signal }) => {
      const cached = getPrintingsByName(card.name);
      if (cached.printings.length > 0 && cached.cachedAt && !isPrintingsStale(cached.cachedAt)) {
        return cached.printings;
      }
      try {
        const fresh = await fetchPrintings(card.name, signal);
        upsertPrintings(card.name, fresh);
        return fresh;
      } catch (err) {
        if (cached.printings.length > 0) return cached.printings;
        throw err;
      }
    },
    enabled: !!card?.name,
    staleTime: 6 * 60 * 60 * 1000, // 6h — prices drift but printings rarely change
    gcTime: DAY_MS,
  });
}
