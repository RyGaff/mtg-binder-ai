import type { CachedCard } from '../db/cards';
import * as db from '../db/cards';
import { fetchCardById } from './scryfall';

/** In-memory cache scoped to the current scan session. Cleared on unmount. */
const sessionCache = new Map<string, CachedCard>();

/** Resolve a card: session cache → fresh SQLite row → Scryfall. Writes through on API hit. */
export async function resolveCardById(scryfallId: string): Promise<CachedCard> {
  const cached = sessionCache.get(scryfallId);
  if (cached) return cached;

  const fromDb = db.getCardById(scryfallId);
  if (fromDb && !db.isCardStale(fromDb)) {
    sessionCache.set(scryfallId, fromDb);
    return fromDb;
  }

  const fresh = await fetchCardById(scryfallId);
  try {
    db.upsertCard(fresh);
  } catch (err) {
    console.warn('[resolveCardById] DB upsert failed, session cache still populated:', err);
  }
  sessionCache.set(scryfallId, fresh);
  return fresh;
}

export function clearSessionCardCache(): void {
  sessionCache.clear();
}

export function getSessionCacheSize(): number {
  return sessionCache.size;
}
