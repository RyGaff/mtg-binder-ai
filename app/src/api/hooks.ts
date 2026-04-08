import { useQuery } from '@tanstack/react-query';
import { fetchCardById, fetchCardByName, searchScryfall } from './scryfall';
import { getCardById, upsertCard, isCardStale, searchCardsLocal } from '../db/cards';

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

/** Build and run a synergy query: extract keywords + oracle fragments from a card. */
export function useSynergySearch(cardName: string) {
  return useQuery({
    queryKey: ['synergy', cardName],
    queryFn: async () => {
      const source = await fetchCardByName(cardName);
      upsertCard(source);
      const keywords: string[] = JSON.parse(source.keywords || '[]');
      const oracleFragments = extractOracleFragments(source.oracle_text);
      const terms = [...keywords.slice(0, 2), ...oracleFragments.slice(0, 3)];
      if (terms.length === 0) return [];
      const query = terms.map(t => `o:"${t}"`).join(' or ');
      const results = await searchScryfall(query);
      results.forEach(upsertCard);
      return results.filter(c => c.scryfall_id !== source.scryfall_id);
    },
    enabled: cardName.length > 1,
  });
}

const ORACLE_PATTERN = /(?:when|whenever|at the beginning of|deals? \d+ damage|draw a? cards?|enters? the battlefield|dies|sacrifice)/gi;

function extractOracleFragments(oracleText: string): string[] {
  const matches = oracleText.match(ORACLE_PATTERN) ?? [];
  return [...new Set(matches.map(m => m.toLowerCase()))];
}
