import { useQuery } from '@tanstack/react-query';
import { fetchCardById, searchScryfall, fetchPrintings } from './scryfall';
import type { PrintingSummary } from './scryfall';
import { getCardById, upsertCard, isCardStale, searchCardsLocal } from '../db/cards';
import type { CachedCard } from '../db/cards';
import { getEmbeddingMap } from '../embeddings/parser';
import { similaritySearch } from '../embeddings/similarity';
import { useStore } from '../store/useStore';
import { fetchEdhrecSynergies, type SynergyResult } from './edhrec';

/** Returns a card, using SQLite cache and falling back to Scryfall. */
export function useCard(scryfallId: string) {
  return useQuery({
    queryKey: ['card', scryfallId],
    queryFn: async () => {
      const cached = getCardById(scryfallId);
      if (cached && !isCardStale(cached)) return cached;
      const fresh = await fetchCardById(scryfallId);
      upsertCard(fresh);
      return fresh;
    },
    staleTime: 24 * 60 * 60 * 1000,
    initialData: () => {
      const cached = getCardById(scryfallId);
      return cached && !isCardStale(cached) ? cached : undefined;
    },
  });
}

/** Search Scryfall and cache results. Falls back to local FTS when offline. */
export function useScryfallSearch(query: string) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: async () => {
      try {
        const results = await searchScryfall(query);
        results.forEach(upsertCard);
        return results;
      } catch {
        return searchCardsLocal(query);
      }
    },
    enabled: query.length > 1,
    staleTime: 5 * 60 * 1000,
  });
}

/** Pull synergy partners from EDHREC (co-occurrence data across public deck lists). */
export function useSynergyFromCard(seed: CachedCard | null) {
  return useQuery<SynergyResult>({
    queryKey: ['synergy', seed?.scryfall_id ?? ''],
    queryFn: () =>
      seed
        ? fetchEdhrecSynergies(seed)
        : Promise.resolve({ metric: 'synergy' as const, entries: [] }),
    enabled: !!seed,
    staleTime: 24 * 60 * 60 * 1000,
    retry: false,
  });
}

// ---------------------------------------------------------------------------
// Effect category extraction (kept — may be used for synergy feature later)
// ---------------------------------------------------------------------------

type EffectCategory = {
  name: string;
  patterns: RegExp[];
};

const EFFECT_CATEGORIES: EffectCategory[] = [
  // deals? covers both "deals" and "deal" forms in MTG oracle text; \bdamage\b is intentionally broad (not "damage to")
  { name: 'damage',   patterns: [/\bdeals?\b/i, /\bdamage\b/i] },
  { name: 'draw',     patterns: [/\bdraw\b.*\bcards?\b/i] },
  { name: 'destroy',  patterns: [/destroy (?:target|all)/i] },
  { name: 'exile',    patterns: [/exile (?:target|all)/i] },
  { name: 'counter',  patterns: [/counter target/i] },
  { name: 'tokens',   patterns: [/\btoken\b/i] },
  // Single regex with | alternation (OR semantics) — not AND like multi-entry categories
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
  return EFFECT_CATEGORIES
    .filter(cat => cat.patterns.every(p => p.test(oracleText)))
    .map(cat => cat.name);
}

export function parseManaValue(manaCost: string): number {
  const tokens = manaCost.match(/\{([^}]+)\}/g) ?? [];
  return tokens.reduce((sum, token) => {
    const inner = token.slice(1, -1);
    const n = parseInt(inner, 10);
    if (!isNaN(n)) return sum + n;
    if (inner === 'X') return sum;
    return sum + 1;
  }, 0);
}

export function buildSimilarQuery(card: CachedCard): string {
  const categories = extractEffectCategories(card.oracle_text);
  const mv = parseManaValue(card.mana_cost);
  const colors = JSON.parse(card.color_identity) as string[];
  const parts: string[] = [];
  if (categories.length > 0) {
    const clauses = categories.map(cat => ORACLE_CLAUSE[cat]).filter(Boolean);
    if (clauses.length === 1) {
      parts.push(clauses[0]);
    } else {
      parts.push(`(${clauses.join(') or (')})`);
    }
  }
  parts.push(`mv>=${Math.max(0, mv - 1)} mv<=${mv + 1}`);
  if (colors.length > 0) parts.push(`c:${colors.join('')}`);
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Embedding-based similar card search
// ---------------------------------------------------------------------------

/** Fetch the top 20 most similar cards using on-device embedding cosine similarity. */
export function useSimilarSearch(card: CachedCard) {
  const embeddingStatus = useStore((s) => s.embeddingStatus);

  return useQuery({
    queryKey: ['similar-embedding', card.scryfall_id],
    queryFn: async () => {
      const { byId, byName } = await getEmbeddingMap();
      console.log('[similar] map sizes — byId:', byId.size, 'byName:', byName.size);
      // The embedding was built from oracle-cards (one printing per unique card).
      // The app may have cached a different printing, so fall back to name lookup.
      const embeddingId = byId.has(card.scryfall_id)
        ? card.scryfall_id
        : byName.get(card.name);
      console.log('[similar] card:', card.name, 'scryfall_id:', card.scryfall_id, 'embeddingId:', embeddingId);
      const results = embeddingId ? similaritySearch(embeddingId, byId, 20) : [];
      console.log('[similar] results count:', results.length);
      const settled = await Promise.allSettled(
        results.map(async ({ scryfallId }) => {
          const cached = getCardById(scryfallId);
          if (cached) return cached;
          const fresh = await fetchCardById(scryfallId);
          upsertCard(fresh);
          return fresh;
        })
      );
      return settled
        .filter((r): r is PromiseFulfilledResult<CachedCard> => r.status === 'fulfilled')
        .map(r => r.value);
    },
    enabled: embeddingStatus === 'idle',
    staleTime: 24 * 60 * 60 * 1000,
  });
}

// ---------------------------------------------------------------------------
// Printings
// ---------------------------------------------------------------------------

export function buildPrintingsQueryKey(card: CachedCard): [string, string] {
  return ['printings', card.name];
}

/** Fetch all printings of a card sorted newest first. Not cached in SQLite. */
export function usePrintings(card: CachedCard) {
  return useQuery({
    queryKey: buildPrintingsQueryKey(card),
    queryFn: () => fetchPrintings(card.name),
    staleTime: 30 * 60 * 1000,
  });
}
