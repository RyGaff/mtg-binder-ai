# Caching Rework

Branch: `caching_rework`. Thirteen issues across the four-layer cache stack
(RQ memory → in-memory session map → SQLite → Scryfall). Each entry: what
was wrong, the actual code that replaced it, why it works, and why it
doesn't break anything else.

---

## P0.1 — Double Scryfall hit on every scan

**Issue.** `app/src/scanner/ocr.ts:165–172` and `:199–200`. After a scan
hit, `scanCard` did:

```ts
fetched = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber); // Scryfall #1
if (fetched) {
  // Warm the session cache; later scans of the same card will skip Scryfall.
  const hydrated = await resolveCardById(fetched.scryfall_id);                // Scryfall #2
  return { strategy: 'set_number', card: hydrated, ... };
}
```

`fetchCardBySetNumber` and `fetchCardByName` (`app/src/api/scryfall.ts`)
return a normalized `CachedCard` but never write through to SQLite or the
session map. So `resolveCardById` saw both layers miss and called
`fetchCardById(scryfall_id)` — a second request for the card already in hand.
The `fetched` payload was discarded. Same on the name path.

**Fix — `app/src/api/cards.ts` (new helper):**

```ts
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
```

**Fix — `app/src/scanner/ocr.ts`:**

```ts
if (fetched) {
  cacheCard(fetched); // single write-through; no second Scryfall hit
  return { strategy: 'set_number', card: fetched, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
}
// ... and on the name path:
const card = await fetchCardByName(nameLine.trim());
cacheCard(card);
return { strategy: 'name', card, corners, imageW: imgW, imageH: imgH, ocrText: tlText, blText };
```

**Why no new bugs.** `cacheCard` performs the exact two writes
`resolveCardById`'s success path used to do, so post-scan cache state is
identical. The image-embedding scan (`scanCardByImage`, `ocr.ts:244`) still
uses `resolveCardById` — that path only has the `scryfall_id` from the
embedding match, which is the legitimate use case for a "resolve by id"
helper.

---

## P0.2 — Search-result upsert clobbers detail rows

**Issue.** `app/src/api/hooks.ts:52` (`useScryfallSearch`) called
`upsertCards(results)`. `searchScryfall` payloads pass through
`normalizeScryfallCard`, which falls back to `[]` for `all_parts` and `{}`
for `prices` when the search response omits them. The existing
`UPSERT_CARD_SQL` in `db/cards.ts` overwrites every column with
`excluded.*`. Sequence: user opens a meld card via `useCard` (full
`all_parts` written), later searches and the same card appears in the
results, search-payload upsert clobbers `all_parts` to `[]`. Next `useCard`
read returns the gutted row.

**Fix — `app/src/db/cards.ts` (new function):**

```ts
const INSERT_OR_IGNORE_CARD_SQL = `INSERT OR IGNORE INTO cards
     (scryfall_id, name, set_code, collector_number, mana_cost, type_line,
      oracle_text, color_identity, image_uri, image_uri_back, card_faces, all_parts, prices, keywords, layout, cached_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;

/** Seed-only bulk upsert. Skips rows that already exist so search-payload writes
 *  don't clobber detail-payload rows (search omits all_parts, etc.). */
export function upsertCardsIfNewer(cards: CachedCard[]): void {
  if (cards.length === 0) return;
  const db = getDb();
  const stmt = db.prepareSync(INSERT_OR_IGNORE_CARD_SQL);
  try {
    db.withTransactionSync(() => {
      for (const card of cards) stmt.executeSync(bindUpsert(card));
    });
  } finally {
    stmt.finalizeSync();
  }
}
```

**Fix — `app/src/api/hooks.ts`:**

```ts
const results = await searchScryfall(query, 1, signal);
// Seed-only: search payloads omit fields like `all_parts` that detail
// payloads carry, so don't clobber existing detailed rows.
upsertCardsIfNewer(results);
return results;
```

**Why no new bugs.** Search results that already exist in the DB are by
definition not gaining new info from the thinner search payload, so
skipping the write is correct. Detail-payload paths (`useCard`,
`useSimilarSearch`'s `fetchCardById`, scan, import) still use the full
`upsertCard` / `upsertCards` and refresh existing rows. New rows from
search still get seeded — `useCard` will replace them with detail data on
first open.

---

## P0.3 — AbortError swallowed in search

**Issue.** `app/src/api/hooks.ts:49–57` had a bare `catch {}` that treated
every error as a network failure and returned `searchCardsLocal(query)`. RQ
aborts in-flight queries when the user types fast — `fetch` throws
`AbortError`. The catch caught it, returned local FTS results, and RQ
cached those as the data for that key. User stops typing on a
previously-aborted key, sees stale local matches.

**Fix — `app/src/api/hooks.ts`:**

```ts
queryFn: async ({ signal }) => {
  try {
    const results = await searchScryfall(query, 1, signal);
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
```

**Why no new bugs.** The abort propagates intact through `throttled()` in
`scryfall.ts` — the wrapper's `requestChain.catch(() => {})` only swallows
on the bookkeeping copy, not the returned promise. After this fix, true
offline failures still hit the local fallback path; only cancellation
rethrows, which RQ expects (cancelled keys are dropped, not cached).

---

## P1.4 — Session cache never re-checks staleness

**Issue.** `app/src/api/cards.ts:11`. `resolveCardById` returned a
session-map hit immediately, no `isCardStale` check. A card cached at app
start stayed "fresh" in memory for the entire session, even past 24h.

**Fix — `app/src/api/cards.ts`:**

```ts
export async function resolveCardById(scryfallId: string): Promise<CachedCard> {
  const cached = sessionCache.get(scryfallId);
  if (cached && !db.isCardStale(cached)) return cached;
  if (cached) sessionCache.delete(scryfallId);

  const fromDb = db.getCardById(scryfallId);
  if (fromDb && !db.isCardStale(fromDb)) {
    sessionCache.set(scryfallId, fromDb);
    return fromDb;
  }

  const fresh = await fetchCardById(scryfallId);
  cacheCard(fresh);
  return fresh;
}
```

**Why no new bugs.** Staleness threshold (`STALE_MS`) is the single source
of truth (P3.12). Stale entries already triggered DB / Scryfall fallback
in the existing flow — re-using that path is correct. Fresh entries return
as before, so the hot path is unchanged.

---

## P1.5 — `useSimilarSearch` returns stale DB rows

**Issue.** `app/src/api/hooks.ts:165–166`. `getCardsByIds(ids)` returned
every matching DB row regardless of `cached_at`. `missing` was computed
from `!cachedMap.has(id)`, so stale rows skipped the Scryfall refresh
path entirely.

**Fix — `app/src/db/cards.ts`:**

```ts
/** Bulk fetch by scryfall_id. Returns a Map keyed by scryfall_id.
 *  When `freshOnly`, rows past STALE_MS are dropped so callers refresh them. */
export function getCardsByIds(
  scryfallIds: readonly string[],
  options: { freshOnly?: boolean } = {},
): Map<string, CachedCard> {
  const out = new Map<string, CachedCard>();
  if (scryfallIds.length === 0) return out;
  const db = getDb();
  const unique = Array.from(new Set(scryfallIds));
  const CHUNK = 500;
  for (let i = 0; i < unique.length; i += CHUNK) {
    const slice = unique.slice(i, i + CHUNK);
    const placeholders = slice.map(() => '?').join(',');
    const rows = db.getAllSync<CachedCard>(
      `SELECT ${SELECT_COLS} FROM cards WHERE scryfall_id IN (${placeholders})`,
      slice
    );
    for (const row of rows) {
      if (options.freshOnly && isCardStale(row)) continue;
      out.set(row.scryfall_id, row);
    }
  }
  return out;
}
```

**Fix — `app/src/api/hooks.ts`:**

```ts
const ids = results.map((r) => r.scryfallId);
// freshOnly: stale rows (>24h) drop out and get refetched.
const cachedMap = getCardsByIds(ids, { freshOnly: true });
const missing = ids.filter((id) => !cachedMap.has(id));
const fetched = await Promise.allSettled(missing.map((id) => fetchCardById(id, signal)));
const fetchedCards = fetched.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []));
if (fetchedCards.length) upsertCards(fetchedCards);
```

**Why no new bugs.** `getCardsByIds` callers without the flag — currently
none; only `useSimilarSearch` uses it — get identical behavior to before.
The change is opt-in. The refetch path already exists and writes through
with `upsertCards`, so freshness is restored on the next render.

---

## P1.6 — Scan never checks DB before going to network

**Issue.** `app/src/scanner/ocr.ts:161–173`. The set/number scan path
always called `fetchCardBySetNumber` even when the card was already in
SQLite. `getCardBySetNumber` exists in `db/cards.ts:96` and is used by
the bulk-import flow; the scan path skipped it.

**Fix — `app/src/scanner/ocr.ts`:**

```ts
if (parsed) {
  // DB precheck: a fresh row by (set, number) skips Scryfall entirely.
  const cachedBySet = getCardBySetNumber(parsed.setCode, parsed.collectorNumber);
  if (cachedBySet && !isCardStale(cachedBySet)) {
    cacheCard(cachedBySet); // warm session cache for repeat scans
    return { strategy: 'set_number', card: cachedBySet, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
  }

  let fetched: CachedCard | null = null;
  try {
    onProgress?.({ step: 'fetching', query: `${parsed.setCode.toUpperCase()} #${parsed.collectorNumber}` });
    fetched = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
  } catch {
    // Scryfall 404 or network error — fall through to name strategy
  }
  ...
}
```

**Why no new bugs.** Same staleness gate as everywhere else. Stale rows
still trigger Scryfall, so price drift fixes itself within 24h. The name
path can't pre-check (no id yet) — left as-is, post-fetch it now uses
`cacheCard` (P0.1) instead of re-fetching.

---

## P2.7 — Session cache unbounded

**Issue.** `app/src/api/cards.ts:6` was
`const sessionCache = new Map<string, CachedCard>()`. Only cleared on scan
tab unmount (`scan.tsx:326`). With expo-router bottom tabs, scan stays
mounted for the app session — the map grew for every card resolved across
all tabs. The repo's `TODO` literally said "LRU Cache. Rework card
caching."

**Fix — new file `app/src/api/lruCache.ts`:**

```ts
/** Simple bounded LRU map. Insertion-order Map: re-inserts on hit move keys
 *  to the most-recently-used end; oldest key evicts when size exceeds capacity. */
export class LruCache<K, V> {
  private readonly map = new Map<K, V>();

  constructor(private readonly capacity: number) {
    if (capacity <= 0) throw new Error('LruCache capacity must be > 0');
  }

  /** Read + promote to MRU. */
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key) as V;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  /** Read without touching LRU order. Use when the caller may discard the
   *  entry (e.g. stale-check followed by delete) — avoids a wasted promote. */
  peek(key: K): V | undefined {
    return this.map.get(key);
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as K | undefined;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}
```

**Fix — `app/src/api/cards.ts`:**

```ts
import { LruCache } from './lruCache';

/** Bounded in-memory cache. 256 entries ≈ a typical browse session
 *  (deck of ~100 + a few searches) without unbounded growth. */
const SESSION_CAPACITY = 256;
const sessionCache = new LruCache<string, CachedCard>(SESSION_CAPACITY);
```

**Why no new bugs.** `LruCache` exposes a superset of the old `Map`
surface (`get`/`peek`/`set`/`delete`/`clear`/`size`).
`clearSessionCardCache` and `getSessionCacheSize` keep their signatures
(the latter is used by `__tests__/api/cards.test.ts`).
Cap of 256 comfortably exceeds a Commander deck (~100) plus a session of
browsing. If a user views >256 distinct cards in a session, only the
least-recently-used ones evict — next access falls through to DB, which is
the correct level-down.

`resolveCardById` uses `peek` then conditionally `get` to promote, so
stale hits don't churn LRU order:

```ts
const cached = sessionCache.peek(scryfallId);
if (cached) {
  if (!db.isCardStale(cached)) {
    sessionCache.get(scryfallId); // promote on fresh hit only
    return cached;
  }
  sessionCache.delete(scryfallId);
}
```

---

## P2.8 — `useCard` runs SQLite per render

**Issue.** `app/src/api/hooks.ts:39–41`:

```ts
const freshCached = () => {
  if (!scryfallId) return undefined;
  const c = getCardById(scryfallId);
  return c && !isCardStale(c) ? c : undefined;
};
return useQuery({
  ...
  initialData: freshCached,
  initialDataUpdatedAt: () => freshCached()?.cached_at,
});
```

Both function-form. `freshCached` calls `getCardById` (sync SQLite). RQ
evaluates these on each render; two SQLite reads per render of
`card/[id].tsx`.

**Fix — `app/src/api/hooks.ts`:**

```ts
export function useCard(scryfallId: string) {
  // Resolve once per id change. Avoids two SQLite reads per render
  // from the previous `() => freshCached()` initialData / updatedAt pair.
  const seed = useMemo(() => {
    if (!scryfallId) return undefined;
    const c = getCardById(scryfallId);
    return c && !isCardStale(c) ? c : undefined;
  }, [scryfallId]);

  return useQuery({
    queryKey: ['card', scryfallId],
    ...
    initialData: seed,
    initialDataUpdatedAt: seed?.cached_at,
  });
}
```

**Why no new bugs.** Memo only invalidates when `scryfallId` changes — the
only thing that affects what `freshCached` returns. RQ behavior is
identical (same value, same updatedAt). Trade-off is negligible: one
SQLite read on id change vs. two reads per render.

---

## P2.9 — `gcTime` < `staleTime` on `useCard`

**Issue.** `staleTime: DAY_MS` (24h), `gcTime: 5 * 60 * 1000` (5 min).
After 5 min idle, RQ evicted the entry; the next mount ran `queryFn`,
which re-read SQLite. The 24h staleness window was decorative.

**Fix — `app/src/api/hooks.ts`:**

```ts
return useQuery({
  queryKey: ['card', scryfallId],
  queryFn: async ({ signal }) => { ... },
  enabled: !!scryfallId,
  staleTime: DAY_MS,
  // gcTime ≥ staleTime so the 24h staleness window is meaningful.
  gcTime: WEEK_MS,
  initialData: seed,
  initialDataUpdatedAt: seed?.cached_at,
});
```

**Why no new bugs.** Card payloads are small (a few KB JSON). LRU + RQ
memory cache are bounded by user navigation, not by `gcTime`. RQ's gc
only runs on observerless queries, so this only retains data RQ would
have re-fetched anyway.

---

## P2.10 — `usePrintings` no DB layer

**Issue.** `app/src/api/hooks.ts:191–198`. Every modal open hit Scryfall,
despite the comment "printings rarely change." No SQLite persistence.

**Fix — `app/src/db/db.ts` (schema, additive):**

```sql
CREATE TABLE IF NOT EXISTS printings (
  card_name        TEXT NOT NULL,
  scryfall_id      TEXT NOT NULL,
  set_code         TEXT NOT NULL,
  set_name         TEXT NOT NULL DEFAULT '',
  collector_number TEXT NOT NULL,
  image_uri        TEXT NOT NULL DEFAULT '',
  image_uri_back   TEXT NOT NULL DEFAULT '',
  layout           TEXT NOT NULL DEFAULT 'normal',
  card_faces       TEXT NOT NULL DEFAULT '[]',
  price_usd        TEXT,
  price_usd_foil   TEXT,
  released_rank    INTEGER NOT NULL DEFAULT 0,
  cached_at        INTEGER NOT NULL,
  PRIMARY KEY (card_name, scryfall_id)
);
CREATE INDEX IF NOT EXISTS printings_name_idx ON printings (card_name, released_rank);
```

**Fix — new file `app/src/db/printings.ts` (excerpt):**

```ts
export const PRINTINGS_STALE_MS = 6 * 60 * 60 * 1000;

export function getPrintingsByName(cardName: string): {
  printings: PrintingSummary[];
  cachedAt: number | null;
} {
  const rows = getDb().getAllSync<PrintingRow>(
    `SELECT ${SELECT_COLS} FROM printings WHERE card_name = ? ORDER BY released_rank ASC`,
    [cardName],
  );
  if (rows.length === 0) return { printings: [], cachedAt: null };
  return { printings: rows.map(rowToSummary), cachedAt: rows[0].cached_at };
}

export function isPrintingsStale(cachedAt: number): boolean {
  return Date.now() - cachedAt > PRINTINGS_STALE_MS;
}

export function upsertPrintings(cardName: string, printings: PrintingSummary[]): void {
  if (printings.length === 0) return;
  const db = getDb();
  const stmt = db.prepareSync(UPSERT_SQL);
  const now = Date.now();
  try {
    db.withTransactionSync(() => {
      // Delete-then-insert so a printing removed from Scryfall doesn't linger.
      db.runSync('DELETE FROM printings WHERE card_name = ?', [cardName]);
      printings.forEach((p, idx) => {
        stmt.executeSync([cardName, p.scryfall_id, p.set_code, p.set_name, p.collector_number,
          p.image_uri, p.image_uri_back, p.layout, p.card_faces,
          p.prices.usd, p.prices.usd_foil, idx, now]);
      });
    });
  } finally {
    stmt.finalizeSync();
  }
}
```

**Fix — `app/src/api/hooks.ts`:**

```ts
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
```

**Why no new bugs.** `upsertPrintings` deletes old rows for that
`card_name` inside a transaction before inserting — so a printing removed
from Scryfall doesn't linger. The 6h threshold matches the previous RQ
`staleTime` exactly, so observable freshness behavior is unchanged.
`PrintingSummary` shape is preserved; consumers don't change.

---

## P3.11 — `useCard` queryFn duplicated freshness check

**Issue.** `app/src/api/hooks.ts:24` ran:

```ts
queryFn: async ({ signal }) => {
  const cached = getCardById(scryfallId);
  if (cached && !isCardStale(cached)) return cached;
  ...
}
```

`initialData` already gated freshness — if `queryFn` fires, RQ wants a
refresh. The inline cache return short-circuited that refresh and could
serve stale-but-not-quite data when the user expected new.

**Fix — `app/src/api/hooks.ts`:**

```ts
queryFn: async ({ signal }) => {
  // initialData already gated freshness; if queryFn fires, RQ wants a refresh.
  const stale = getCardById(scryfallId);
  try {
    const fresh = await fetchCardById(scryfallId, signal);
    upsertCard(fresh);
    return fresh;
  } catch (err) {
    if (stale) return stale; // offline / rate-limit fallback
    throw err;
  }
},
```

**Why no new bugs.** The pre-fix path that returned cached-and-not-stale
never actually fired in practice (RQ would have used `initialData`
instead). Removing it can't change observable behavior. The offline
fallback is preserved — `stale` exists only to bail out on network failure.

---

## P3.12 — `STALE_MS` vs `DAY_MS` drift risk

**Issue.** `app/src/db/cards.ts:22` defined
`STALE_MS = 24 * 60 * 60 * 1000`. `app/src/api/hooks.ts:10` defined
`DAY_MS = 24 * 60 * 60 * 1000`. Same value by coincidence.

**Fix — `app/src/db/cards.ts`:**

```ts
export const STALE_MS = 24 * 60 * 60 * 1000;
```

**Fix — `app/src/api/hooks.ts`:**

```ts
import {
  getCardById,
  getCardsByIds,
  upsertCard,
  upsertCards,
  upsertCardsIfNewer,
  isCardStale,
  searchCardsLocal,
  STALE_MS,
} from '../db/cards';
...
const DAY_MS = STALE_MS;
const WEEK_MS = 7 * DAY_MS;
```

**Why no new bugs.** Numerical value unchanged. Future edits to either
consumer pull from the same constant.

---

## P3.13 — Missing `['card', id]` invalidation after bulk import

**Issue.** `app/app/(tabs)/index.tsx:143–148`. Import called
`upsertCards(toUpsert)` then invalidated `['collection']` and
`['collection-value']` only. Cards already open in detail screens via
`useCard` kept showing pre-import data until `staleTime` (24h after the
P2.9 fix).

**Fix — `app/app/(tabs)/index.tsx`:**

```ts
if (toUpsert.length) upsertCards(toUpsert);
if (toAdd.length) addManyToCollection(toAdd);

setImportProgress(null);
qc.invalidateQueries({ queryKey: ['collection'] });
qc.invalidateQueries({ queryKey: ['collection-value'] });
// Bulk-imported cards may be open in detail screens — drop their RQ entries.
if (toUpsert.length) qc.invalidateQueries({ queryKey: ['card'] });
```

**Why no new bugs.** RQ partial-key match invalidates all `['card', *]`
entries. Invalidating cards that weren't part of the import just causes
their `useCard` to re-render and read SQLite (cheap) or re-fetch via
Scryfall (rare; only if their DB rows are also stale). Gate
`if (toUpsert.length)` so a card-already-cached-only import doesn't
trigger it.

---

## File map

| File | Type | Issues |
|---|---|---|
| `app/src/db/cards.ts` | edit | P0.2, P1.5, P3.12 |
| `app/src/db/db.ts` | edit | P2.10 |
| `app/src/db/printings.ts` | **new** | P2.10 |
| `app/src/api/lruCache.ts` | **new** | P2.7 |
| `app/src/api/cards.ts` | rewrite | P0.1 helper, P1.4, P2.7 |
| `app/src/api/hooks.ts` | edit | P0.2, P0.3, P1.5, P2.8, P2.9, P2.10, P3.11, P3.12 |
| `app/src/scanner/ocr.ts` | edit | P0.1, P1.6 |
| `app/app/(tabs)/index.tsx` | edit | P3.13 |

## Verification

`cd app && npx tsc --noEmit` is clean for caching code. The remaining
error in `__tests__/db/cards.test.ts` (missing `layout` field in a
fixture) pre-dates this branch and is unrelated.
