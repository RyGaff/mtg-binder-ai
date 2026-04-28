import type { CachedCard } from '../db/cards';
import * as db from '../db/cards';
import { fetchCardById } from './scryfall';
import { LruCache } from './lruCache';

/** Bounded in-memory cache for card JSON (~2 KB per entry). Capacity-only —
 *  capping at 256 puts the whole cache at ~500 KB max. This is a latency
 *  optimization (skip SQLite reads on repeat lookups), not a memory bound. */
const SESSION_CAPACITY = 256;
const sessionCache = new LruCache<string, CachedCard>(SESSION_CAPACITY);

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

/** Session-only setter. Use when the row already lives on disk (e.g. a fresh
 *  SQLite read) — avoids a wasted upsert that `cacheCard` would do. */
export function setSessionCard(card: CachedCard): void {
  sessionCache.set(card.scryfall_id, card);
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
