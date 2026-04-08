import { getDb } from './db';

export type CachedCard = {
  scryfall_id: string;
  name: string;
  set_code: string;
  collector_number: string;
  mana_cost: string;
  type_line: string;
  oracle_text: string;
  color_identity: string; // JSON string
  image_uri: string;
  prices: string; // JSON string
  keywords: string; // JSON string
  cached_at: number;
};

const STALE_MS = 24 * 60 * 60 * 1000; // 24 hours

const SELECT_COLS = 'scryfall_id, name, set_code, collector_number, mana_cost, type_line, oracle_text, color_identity, image_uri, prices, keywords, cached_at';

export function upsertCard(card: CachedCard): void {
  const db = getDb();
  db.runSync(
    `INSERT INTO cards
       (scryfall_id, name, set_code, collector_number, mana_cost, type_line,
        oracle_text, color_identity, image_uri, prices, keywords, cached_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(scryfall_id) DO UPDATE SET
       name=excluded.name, set_code=excluded.set_code,
       collector_number=excluded.collector_number, mana_cost=excluded.mana_cost,
       type_line=excluded.type_line, oracle_text=excluded.oracle_text,
       color_identity=excluded.color_identity, image_uri=excluded.image_uri,
       prices=excluded.prices, keywords=excluded.keywords,
       cached_at=excluded.cached_at`,
    [
      card.scryfall_id, card.name, card.set_code.toLowerCase(), card.collector_number,
      card.mana_cost, card.type_line, card.oracle_text, card.color_identity,
      card.image_uri, card.prices, card.keywords, card.cached_at,
    ]
  );
}

export function getCardById(scryfallId: string): CachedCard | null {
  const db = getDb();
  return db.getFirstSync<CachedCard>(
    `SELECT ${SELECT_COLS} FROM cards WHERE scryfall_id = ?`,
    [scryfallId]
  ) ?? null;
}

export function getCardBySetNumber(setCode: string, collectorNumber: string): CachedCard | null {
  const db = getDb();
  return db.getFirstSync<CachedCard>(
    `SELECT ${SELECT_COLS} FROM cards WHERE set_code = ? AND collector_number = ?`,
    [setCode.toLowerCase(), collectorNumber]
  ) ?? null;
}

export function isCardStale(card: CachedCard): boolean {
  return Date.now() - card.cached_at > STALE_MS;
}

export function searchCardsLocal(query: string, limit = 20): CachedCard[] {
  const db = getDb();
  return db.getAllSync<CachedCard>(
    `SELECT c.scryfall_id, c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.prices, c.keywords, c.cached_at
     FROM cards_fts
     JOIN cards c ON c.scryfall_id = cards_fts.scryfall_id
     WHERE cards_fts MATCH ?
     LIMIT ?`,
    [`"${query.replace(/"/g, '""')}"*`, limit]
  );
}
