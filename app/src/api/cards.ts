import type { CachedCard } from '../db/cards';
import * as db from '../db/cards';
import { fetchCardById } from './scryfall';
import { LruCache } from './lruCache';

/** Bounded in-memory cache. 256 entries ≈ a typical browse session
 *  (deck of ~100 + a few searches). 15-min idle TTL means entries the user
 *  hasn't touched recently fall out before they pile up against the cap.
 *  Both bounds matter: capacity for active churn, TTL for idle accumulation. */
const SESSION_CAPACITY = 256;
const SESSION_TTL_MS = 15 * 60 * 1000;
const sessionCache = new LruCache<string, CachedCard>(SESSION_CAPACITY, SESSION_TTL_MS);

/** Write-through helper. Use after a Scryfall fetch to populate both layers
 *  in one place — replaces the old "fetch, then resolveCardById to warm the cache"
 *  anti-pattern that triggered a second Scryfall request for the same id. */
export function cacheCard(card: CachedCard): void {
  try {
    db.upsertCard(card);
  } catch (err) {
    console.warn('[cacheCard] DB upsert failed, session cache still populated:', err);
  }
  sessionCache.set(card.scryfall_id, card);
}

/** Bulk write-through: SQLite upsert in one transaction, then warm the LRU
 *  per card so subsequent single-card reads hit memory. */
export function cacheCards(cards: CachedCard[]): void {
  if (cards.length === 0) return;
  try {
    db.upsertCards(cards);
  } catch (err) {
    console.warn('[cacheCards] DB bulk upsert failed, session cache still populated:', err);
  }
  for (const card of cards) sessionCache.set(card.scryfall_id, card);
}

/** Read-only LRU peek for callers that already have their own freshness gate. */
export function peekSessionCard(scryfallId: string): CachedCard | undefined {
  return sessionCache.peek(scryfallId);
}

/** Promote an existing entry to MRU. Use when a peek hit is being returned. */
export function touchSessionCard(scryfallId: string): void {
  sessionCache.get(scryfallId);
}

/** Resolve a card: session cache → fresh SQLite row → Scryfall. Writes through on API hit. */
export async function resolveCardById(scryfallId: string): Promise<CachedCard> {
  // peek first — on stale hit we'd just delete the promoted entry.
  const cached = sessionCache.peek(scryfallId);
  if (cached) {
    if (!db.isCardStale(cached)) {
      sessionCache.get(scryfallId); // promote on fresh hit only
      return cached;
    }
    sessionCache.delete(scryfallId);
  }

  const fromDb = db.getCardById(scryfallId);
  if (fromDb && !db.isCardStale(fromDb)) {
    sessionCache.set(scryfallId, fromDb);
    return fromDb;
  }

  const fresh = await fetchCardById(scryfallId);
  cacheCard(fresh);
  return fresh;
}

export function clearSessionCardCache(): void {
  sessionCache.clear();
}

export function getSessionCacheSize(): number {
  return sessionCache.size;
}
