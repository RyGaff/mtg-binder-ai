import { getDb } from './db';
import type { PrintingSummary } from '../api/scryfall';

export const PRINTINGS_STALE_MS = 6 * 60 * 60 * 1000;

const SELECT_COLS =
  'scryfall_id, set_code, set_name, collector_number, image_uri, image_uri_back, layout, card_faces, price_usd, price_usd_foil, released_rank, cached_at';

const UPSERT_SQL = `INSERT INTO printings
    (card_name, scryfall_id, set_code, set_name, collector_number, image_uri,
     image_uri_back, layout, card_faces, price_usd, price_usd_foil, released_rank, cached_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(card_name, scryfall_id) DO UPDATE SET
     set_code=excluded.set_code, set_name=excluded.set_name,
     collector_number=excluded.collector_number,
     image_uri=excluded.image_uri, image_uri_back=excluded.image_uri_back,
     layout=excluded.layout, card_faces=excluded.card_faces,
     price_usd=excluded.price_usd, price_usd_foil=excluded.price_usd_foil,
     released_rank=excluded.released_rank, cached_at=excluded.cached_at`;

type PrintingRow = {
  scryfall_id: string;
  set_code: string;
  set_name: string;
  collector_number: string;
  image_uri: string;
  image_uri_back: string;
  layout: string;
  card_faces: string;
  price_usd: string | null;
  price_usd_foil: string | null;
  released_rank: number;
  cached_at: number;
};

function rowToSummary(r: PrintingRow): PrintingSummary {
  return {
    scryfall_id: r.scryfall_id,
    set_code: r.set_code,
    set_name: r.set_name,
    collector_number: r.collector_number,
    image_uri: r.image_uri,
    image_uri_back: r.image_uri_back,
    layout: r.layout,
    card_faces: r.card_faces,
    prices: { usd: r.price_usd, usd_foil: r.price_usd_foil },
  };
}

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
        stmt.executeSync([
          cardName,
          p.scryfall_id,
          p.set_code,
          p.set_name,
          p.collector_number,
          p.image_uri,
          p.image_uri_back,
          p.layout,
          p.card_faces,
          p.prices.usd,
          p.prices.usd_foil,
          idx,
          now,
        ]);
      });
    });
  } finally {
    stmt.finalizeSync();
  }
}
