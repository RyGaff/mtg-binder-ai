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
  image_uri_back: string;
  card_faces: string; // JSON string — array of {name, mana_cost, type_line, oracle_text, image_uri}, empty [] for single-face
  all_parts: string; // JSON string — meld_part / meld_result entries only
  prices: string; // JSON string
  keywords: string; // JSON string
  cached_at: number;
};

const STALE_MS = 24 * 60 * 60 * 1000;

const SELECT_COLS = 'scryfall_id, name, set_code, collector_number, mana_cost, type_line, oracle_text, color_identity, image_uri, image_uri_back, card_faces, all_parts, prices, keywords, cached_at';

const UPSERT_CARD_SQL = `INSERT INTO cards
     (scryfall_id, name, set_code, collector_number, mana_cost, type_line,
      oracle_text, color_identity, image_uri, image_uri_back, card_faces, all_parts, prices, keywords, cached_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(scryfall_id) DO UPDATE SET
     name=excluded.name, set_code=excluded.set_code,
     collector_number=excluded.collector_number, mana_cost=excluded.mana_cost,
     type_line=excluded.type_line, oracle_text=excluded.oracle_text,
     color_identity=excluded.color_identity, image_uri=excluded.image_uri,
     image_uri_back=excluded.image_uri_back,
     card_faces=excluded.card_faces,
     all_parts=excluded.all_parts,
     prices=excluded.prices, keywords=excluded.keywords,
     cached_at=excluded.cached_at`;

function bindUpsert(card: CachedCard): (string | number)[] {
  return [
    card.scryfall_id, card.name, card.set_code.toLowerCase(), card.collector_number,
    card.mana_cost, card.type_line, card.oracle_text, card.color_identity,
    card.image_uri, card.image_uri_back ?? '', card.card_faces ?? '[]',
    card.all_parts ?? '[]', card.prices, card.keywords, card.cached_at,
  ];
}

export function upsertCard(card: CachedCard): void {
  getDb().runSync(UPSERT_CARD_SQL, bindUpsert(card));
}

/** Bulk upsert in a single transaction with a reused prepared statement. ~10-50x faster on 100+ rows. */
export function upsertCards(cards: CachedCard[]): void {
  if (cards.length === 0) return;
  const db = getDb();
  const stmt = db.prepareSync(UPSERT_CARD_SQL);
  try {
    db.withTransactionSync(() => {
      for (const card of cards) stmt.executeSync(bindUpsert(card));
    });
  } finally {
    stmt.finalizeSync();
  }
}

export function getCardById(scryfallId: string): CachedCard | null {
  return getDb().getFirstSync<CachedCard>(
    `SELECT ${SELECT_COLS} FROM cards WHERE scryfall_id = ?`,
    [scryfallId]
  ) ?? null;
}

/** Bulk fetch by scryfall_id. Returns a Map keyed by scryfall_id. */
export function getCardsByIds(scryfallIds: readonly string[]): Map<string, CachedCard> {
  const out = new Map<string, CachedCard>();
  if (scryfallIds.length === 0) return out;
  const db = getDb();
  // De-dupe and chunk to stay well under SQLITE_MAX_VARIABLE_NUMBER (999).
  const unique = Array.from(new Set(scryfallIds));
  const CHUNK = 500;
  for (let i = 0; i < unique.length; i += CHUNK) {
    const slice = unique.slice(i, i + CHUNK);
    const placeholders = slice.map(() => '?').join(',');
    const rows = db.getAllSync<CachedCard>(
      `SELECT ${SELECT_COLS} FROM cards WHERE scryfall_id IN (${placeholders})`,
      slice
    );
    for (const row of rows) out.set(row.scryfall_id, row);
  }
  return out;
}

export function getCardBySetNumber(setCode: string, collectorNumber: string): CachedCard | null {
  return getDb().getFirstSync<CachedCard>(
    `SELECT ${SELECT_COLS} FROM cards WHERE set_code = ? AND collector_number = ?`,
    [setCode.toLowerCase(), collectorNumber]
  ) ?? null;
}

export function isCardStale(card: CachedCard): boolean {
  return Date.now() - card.cached_at > STALE_MS;
}

export function searchCardsLocal(query: string, limit = 20): CachedCard[] {
  // Joining on rowid is faster than on TEXT scryfall_id — cards_fts uses content=cards so rowids match 1:1.
  return getDb().getAllSync<CachedCard>(
    `SELECT c.scryfall_id, c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.image_uri_back, c.card_faces, c.all_parts, c.prices, c.keywords, c.cached_at
     FROM cards_fts
     JOIN cards c ON c.rowid = cards_fts.rowid
     WHERE cards_fts MATCH ?
     LIMIT ?`,
    [`"${query.replace(/"/g, '""')}"*`, limit]
  );
}
