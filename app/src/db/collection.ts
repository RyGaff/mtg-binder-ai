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

export function addToCollection(args: AddToCollectionArgs): void {
  const db = getDb();
  db.runSync(
    `INSERT INTO collection_entries (scryfall_id, quantity, foil, condition, added_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(scryfall_id, foil, condition)
     DO UPDATE SET quantity = quantity + excluded.quantity`,
    [args.scryfall_id, args.quantity, args.foil ? 1 : 0, args.condition, Date.now()]
  );
}

function normalizeFoil(rows: CollectionEntryWithCard[]): CollectionEntryWithCard[] {
  return rows.map((r) => ({ ...r, foil: !!r.foil }));
}

export function getCollection(): CollectionEntryWithCard[] {
  const db = getDb();
  return normalizeFoil(db.getAllSync<CollectionEntryWithCard>(
    `SELECT ce.id, ce.scryfall_id, ce.quantity, ce.foil, ce.condition, ce.added_at,
            c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.prices, c.keywords, c.cached_at
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id
     ORDER BY c.name ASC`
  ));
}

export function getCollectionByColor(colorCode: string): CollectionEntryWithCard[] {
  const db = getDb();
  return normalizeFoil(db.getAllSync<CollectionEntryWithCard>(
    `SELECT ce.id, ce.scryfall_id, ce.quantity, ce.foil, ce.condition, ce.added_at,
            c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.prices, c.keywords, c.cached_at
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id
     WHERE c.color_identity LIKE ?
     ORDER BY c.name ASC`,
    [`%"${colorCode}"%`]
  ));
}

export function searchCollection(query: string): CollectionEntryWithCard[] {
  const db = getDb();
  return normalizeFoil(db.getAllSync<CollectionEntryWithCard>(
    `SELECT ce.id, ce.scryfall_id, ce.quantity, ce.foil, ce.condition, ce.added_at,
            c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.prices, c.keywords, c.cached_at
     FROM collection_entries ce
     JOIN cards c ON c.scryfall_id = ce.scryfall_id
     WHERE c.name LIKE ?
     ORDER BY c.name ASC`,
    [`%${query}%`]
  ));
}

export function updateQuantity(entryId: number, quantity: number): void {
  const db = getDb();
  db.runSync('UPDATE collection_entries SET quantity = ? WHERE id = ?', [quantity, entryId]);
}

export function removeFromCollection(entryId: number): void {
  const db = getDb();
  db.runSync('DELETE FROM collection_entries WHERE id = ?', [entryId]);
}

export function clearCollection(): void {
  const db = getDb();
  db.runSync('DELETE FROM collection_entries');
}

export function getCollectionTotalValue(): number {
  const db = getDb();
  const rows = db.getAllSync<{ prices: string; quantity: number; foil: number }>(
    'SELECT c.prices, ce.quantity, ce.foil FROM collection_entries ce JOIN cards c ON c.scryfall_id = ce.scryfall_id'
  );
  return rows.reduce((sum, row) => {
    const prices = JSON.parse(row.prices ?? '{}');
    const price = row.foil ? parseFloat(prices.usd_foil ?? '0') : parseFloat(prices.usd ?? '0');
    return sum + price * row.quantity;
  }, 0);
}

export function getFoilCount(): number {
  const db = getDb();
  const row = db.getFirstSync<{ total: number | null }>(
    'SELECT SUM(quantity) as total FROM collection_entries WHERE foil = 1'
  );
  return row?.total ?? 0;
}

export function getTotalCardCount(): number {
  const db = getDb();
  const row = db.getFirstSync<{ total: number | null }>(
    'SELECT SUM(quantity) as total FROM collection_entries'
  );
  return row?.total ?? 0;
}
