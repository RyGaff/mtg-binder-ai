import type { CachedCard } from '../db/cards';
import * as db from '../db/cards';
import { fetchCardById } from './scryfall';

/**
 * In-memory cache scoped to the current scan session. Cleared on
 * scan-screen unmount. A successful Scryfall miss hydrates both this
 * cache and the DB, so a subsequent scan of the same card is free.
 */
const sessionCache: Map<string, CachedCard> = new Map();

/**
 * Resolve a full card record given a Scryfall id. Tries:
 *   1. Session cache  (~1 ms)
 *   2. Local SQLite   (~5 ms) — only if row is fresh
 *   3. Scryfall API   (~200 ms, requires network)
 * On Scryfall hit, writes through to both the DB and the session cache.
 */
export async function resolveCardById(scryfallId: string): Promise<CachedCard> {
  const cached = sessionCache.get(scryfallId);
  if (cached) return cached;

  const fromDb = db.getCardById(scryfallId);
  if (fromDb && !db.isCardStale(fromDb)) {
    sessionCache.set(scryfallId, fromDb);
    return fromDb;
  }

  const fresh = await fetchCardById(scryfallId);
  db.upsertCard(fresh);
  sessionCache.set(scryfallId, fresh);
  return fresh;
}

/** Clear the scan-session cache. Call from the Scan screen's unmount effect. */
export function clearSessionCardCache(): void {
  sessionCache.clear();
}

/** Test/debug hook. */
export function getSessionCacheSize(): number {
  return sessionCache.size;
}
