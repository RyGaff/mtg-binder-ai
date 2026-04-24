import { getDb } from './db';
import type { CachedCard } from './cards';

export type CollectionEntry = {
  id: number;
  scryfall_id: string;
  quantity: number;
  foil: boolean;
  condition: 'NM' | 'LP' | 'MP' | 'HP' | 'DMG';
  added_at: number;
};

export type CollectionEntryWithCard = CollectionEntry & CachedCard;

export type AddToCollectionArgs = {
  scryfall_id: string;
  quantity: number;
  foil: boolean;
  condition: CollectionEntry['condition'];
};

const ADD_TO_COLLECTION_SQL =
  `INSERT INTO collection_entries (scryfall_id, quantity, foil, condition, added_at)
   VALUES (?, ?, ?, ?, ?)
   ON CONFLICT(scryfall_id, foil, condition)
   DO UPDATE SET quantity = quantity + excluded.quantity`;

const ENTRY_WITH_CARD_COLS =
  `ce.id, ce.scryfall_id, ce.quantity, ce.foil, ce.condition, ce.added_at,
   c.name, c.set_code, c.collector_number, c.mana_cost,
   c.type_line, c.oracle_text, c.color_identity, c.image_uri,
   c.image_uri_back, c.card_faces, c.all_parts, c.prices, c.keywords, c.cached_at`;

export function addToCollection(args: AddToCollectionArgs): void {
  getDb().runSync(
    ADD_TO_COLLECTION_SQL,
    [args.scryfall_id, args.quantity, args.foil ? 1 : 0, args.condition, Date.now()]
  );
}

/** Bulk add inside a single transaction with a reused prepared statement. */
export function addManyToCollection(entries: readonly AddToCollectionArgs[]): void {
  if (entries.length === 0) return;
  const db = getDb();
  const now = Date.now();
  const stmt = db.prepareSync(ADD_TO_COLLECTION_SQL);
  try {
    db.withTransactionSync(() => {
      entries.forEach((a, i) => {
        stmt.executeSync([a.scryfall_id, a.quantity, a.foil ? 1 : 0, a.condition, now + i]);
      });
    });
  } finally {
    stmt.finalizeSync();
  }
}

function normalizeFoil(rows: CollectionEntryWithCard[]): CollectionEntryWithCard[] {
  return rows.map((r) => ({ ...r, foil: !!r.foil }));
}

function queryEntries(whereAndOrder: string, params: (string | number)[] = []): CollectionEntryWithCard[] {
  return normalizeFoil(getDb().getAllSync<CollectionEntryWithCard>(
    `SELECT ${ENTRY_WITH_CARD_COLS}
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id
     ${whereAndOrder}`,
    params
  ));
}

export function getCollection(): CollectionEntryWithCard[] {
  return queryEntries('ORDER BY c.name ASC');
}

export function getCollectionByColor(colorCode: string): CollectionEntryWithCard[] {
  return queryEntries('WHERE c.color_identity LIKE ? ORDER BY c.name ASC', [`%"${colorCode}"%`]);
}

function buildFtsNameQuery(raw: string): string | null {
  const cleaned = raw.replace(/["*()^:]/g, ' ').replace(/[-+]/g, ' ').trim();
  if (!cleaned) return null;
  const tokens = cleaned.split(/\s+/).filter(Boolean);
  return tokens.length ? tokens.map((t) => `name : "${t}"*`).join(' AND ') : null;
}

export function searchCollection(query: string): CollectionEntryWithCard[] {
  const trimmed = query.trim();
  if (!trimmed) return getCollection();
  const ftsQuery = buildFtsNameQuery(trimmed);
  if (!ftsQuery) return getCollection();
  return normalizeFoil(getDb().getAllSync<CollectionEntryWithCard>(
    `SELECT ${ENTRY_WITH_CARD_COLS}
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id
     JOIN cards_fts f ON f.rowid = c.rowid
     WHERE cards_fts MATCH ?
     ORDER BY c.name ASC`,
    [ftsQuery]
  ));
}

export function updateQuantity(entryId: number, quantity: number): void {
  getDb().runSync('UPDATE collection_entries SET quantity = ? WHERE id = ?', [quantity, entryId]);
}

export function removeFromCollection(entryId: number): void {
  getDb().runSync('DELETE FROM collection_entries WHERE id = ?', [entryId]);
}

export function clearCollection(): void {
  getDb().runSync('DELETE FROM collection_entries');
}

export function getCollectionTotalValue(): number {
  // Pushes JSON extraction + SUM into SQLite (JSON1 is built into expo-sqlite).
  const row = getDb().getFirstSync<{ total: number | null }>(
    `SELECT SUM(
       CAST(
         COALESCE(
           json_extract(c.prices, CASE WHEN ce.foil = 1 THEN '$.usd_foil' ELSE '$.usd' END),
           '0'
         ) AS REAL
       ) * ce.quantity
     ) AS total
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id`
  );
  return row?.total ?? 0;
}

export function getFoilCount(): number {
  const row = getDb().getFirstSync<{ total: number | null }>(
    'SELECT SUM(quantity) as total FROM collection_entries WHERE foil = 1'
  );
  return row?.total ?? 0;
}

export function getTotalCardCount(): number {
  const row = getDb().getFirstSync<{ total: number | null }>(
    'SELECT SUM(quantity) as total FROM collection_entries'
  );
  return row?.total ?? 0;
}
