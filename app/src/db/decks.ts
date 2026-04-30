import { getDb } from './db';
import type { CachedCard } from './cards';
import { fetchArtCrop } from '../api/scryfall';

export type Board = 'main' | 'side' | 'commander' | 'considering';

export type Deck = {
  id: number;
  name: string;
  format: string;
  created_at: number;
  art_crop_uri: string;
};

export type DeckCard = {
  deck_id: number;
  scryfall_id: string;
  quantity: number;
  board: Board;
} & CachedCard;

export type AddCardToDeckArgs = {
  deck_id: number;
  scryfall_id: string;
  quantity: number;
  board: Board;
};

export function createDeck(args: { name: string; format: string }): number {
  const result = getDb().runSync(
    'INSERT INTO decks (name, format, created_at) VALUES (?, ?, ?)',
    [args.name, args.format, Date.now()]
  );
  return result.lastInsertRowId;
}

export function getDecks(): Deck[] {
  return getDb().getAllSync<Deck>('SELECT id, name, format, created_at, art_crop_uri FROM decks ORDER BY created_at DESC');
}

export function getDeck(deckId: number): Deck | null {
  return getDb().getFirstSync<Deck>(
    'SELECT id, name, format, created_at, art_crop_uri FROM decks WHERE id = ?',
    [deckId],
  );
}

export function renameDeck(deckId: number, name: string): void {
  getDb().runSync('UPDATE decks SET name = ? WHERE id = ?', [name, deckId]);
}

export function setDeckArt(deckId: number, artCropUri: string): void {
  getDb().runSync('UPDATE decks SET art_crop_uri = ? WHERE id = ?', [artCropUri, deckId]);
}

/**
 * Set the deck's art_crop_uri from a card if not already set.
 * Async — fetches art_crop from Scryfall (we don't store it on every cached card).
 */
export async function ensureDeckArt(deckId: number, scryfallId: string): Promise<void> {
  const deck = getDeck(deckId);
  if (!deck || deck.art_crop_uri) return;
  try {
    const uri = await fetchArtCrop(scryfallId);
    if (uri) setDeckArt(deckId, uri);
  } catch { /* network errors / aborts: silently skip */ }
}

export type DeckWithMeta = Deck & {
  card_count: number;
  // Per-board breakdown for the deck-list row chrome. main_count includes the
  // commander board (it's part of the deck proper for Commander-format play),
  // side_count is the sideboard only.
  main_count: number;
  side_count: number;
  color_identity: string[];
};

type DeckMetaRow = {
  id: number;
  name: string;
  format: string;
  created_at: number;
  art_crop_uri: string;
  card_count: number | null;
  main_count: number | null;
  side_count: number | null;
  color_identity_concat: string | null;
};

export function getDecksWithMeta(): DeckWithMeta[] {
  const rows = getDb().getAllSync<DeckMetaRow>(`
    SELECT
      d.id, d.name, d.format, d.created_at, d.art_crop_uri,
      COALESCE(SUM(dc.quantity), 0) AS card_count,
      COALESCE(SUM(CASE WHEN dc.board IN ('main', 'commander') THEN dc.quantity ELSE 0 END), 0) AS main_count,
      COALESCE(SUM(CASE WHEN dc.board = 'side' THEN dc.quantity ELSE 0 END), 0) AS side_count,
      GROUP_CONCAT(c.color_identity, '|') AS color_identity_concat
    FROM decks d
    LEFT JOIN deck_cards dc ON dc.deck_id = d.id
    LEFT JOIN cards c ON c.scryfall_id = dc.scryfall_id
    GROUP BY d.id
    ORDER BY d.created_at DESC
  `);
  return rows.map((r) => {
    const colors = new Set<string>();
    if (r.color_identity_concat) {
      for (const chunk of r.color_identity_concat.split('|')) {
        try {
          const arr = JSON.parse(chunk) as string[];
          for (const c of arr) if ('WUBRG'.includes(c)) colors.add(c);
        } catch { /* skip */ }
      }
    }
    return {
      id: r.id,
      name: r.name,
      format: r.format,
      created_at: r.created_at,
      art_crop_uri: r.art_crop_uri,
      card_count: r.card_count ?? 0,
      main_count: r.main_count ?? 0,
      side_count: r.side_count ?? 0,
      color_identity: Array.from(colors),
    };
  });
}

export function getDeckCards(deckId: number): DeckCard[] {
  return getDb().getAllSync<DeckCard>(
    `SELECT dc.deck_id, dc.scryfall_id, dc.quantity, dc.board,
            c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.image_uri_back, c.card_faces, c.all_parts, c.prices, c.keywords, c.cached_at
     FROM deck_cards dc
     JOIN cards c ON c.scryfall_id = dc.scryfall_id
     WHERE dc.deck_id = ?
     ORDER BY dc.board, c.mana_cost, c.name`,
    [deckId]
  );
}

// Rolling cap on per-deck history rows. Old events past this point are pruned
// after each insert so the table can't grow unbounded.
const HISTORY_RETENTION = 10;

type LogArgs = {
  deck_id: number;
  scryfall_id: string;
  event_type: 'add' | 'remove' | 'decrement' | 'move';
  board_from: Board;
  board_to?: Board;
  qty_delta: number;
};
function logDeckEvent(args: LogArgs): void {
  const db = getDb();
  // Resolve card name once at write time and store it on the history row, so
  // reads are a single-table scan and old rows survive cards-cache eviction.
  const nameRow = db.getFirstSync<{ name: string }>(
    'SELECT name FROM cards WHERE scryfall_id = ?',
    [args.scryfall_id]
  );
  db.runSync(
    `INSERT INTO deck_history (deck_id, scryfall_id, card_name, event_type, board_from, board_to, qty_delta, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      args.deck_id, args.scryfall_id, nameRow?.name ?? '', args.event_type,
      args.board_from, args.board_to ?? null, args.qty_delta, Date.now(),
    ]
  );
  // Prune everything past the retention window for this deck. Doing it inline
  // keeps the table lean without a background job; cost is one extra DELETE
  // per write, bounded by the index on (deck_id, created_at DESC).
  db.runSync(
    `DELETE FROM deck_history
     WHERE deck_id = ? AND id NOT IN (
       SELECT id FROM deck_history WHERE deck_id = ? ORDER BY created_at DESC LIMIT ?
     )`,
    [args.deck_id, args.deck_id, HISTORY_RETENTION]
  );
}

export function addCardToDeck(args: AddCardToDeckArgs): void {
  getDb().runSync(
    `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
    [args.deck_id, args.scryfall_id, args.quantity, args.board]
  );
  logDeckEvent({
    deck_id: args.deck_id, scryfall_id: args.scryfall_id, event_type: 'add',
    board_from: args.board, qty_delta: args.quantity,
  });
}

export function removeCardFromDeck(deckId: number, scryfallId: string, board: Board): void {
  // Capture qty before delete so the history row records how many copies
  // disappeared; useful when re-adding, and informative in the timeline.
  const row = getDb().getFirstSync<{ quantity: number }>(
    'SELECT quantity FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
    [deckId, scryfallId, board]
  );
  if (!row) return;
  getDb().runSync(
    'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
    [deckId, scryfallId, board]
  );
  logDeckEvent({
    deck_id: deckId, scryfall_id: scryfallId, event_type: 'remove',
    board_from: board, qty_delta: row.quantity,
  });
}

/** Decrement qty by 1; deletes the row if qty drops to 0. */
export function decrementCardInDeck(deckId: number, scryfallId: string, board: Board): void {
  const db = getDb();
  // Read current qty so we don't log a phantom decrement on a missing row,
  // and we know whether this drop actually deleted the row.
  const row = db.getFirstSync<{ quantity: number }>(
    'SELECT quantity FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
    [deckId, scryfallId, board]
  );
  if (!row) return;
  db.runSync(
    `UPDATE deck_cards SET quantity = quantity - 1
     WHERE deck_id = ? AND scryfall_id = ? AND board = ?`,
    [deckId, scryfallId, board]
  );
  db.runSync(
    `DELETE FROM deck_cards
     WHERE deck_id = ? AND scryfall_id = ? AND board = ? AND quantity <= 0`,
    [deckId, scryfallId, board]
  );
  logDeckEvent({
    deck_id: deckId, scryfall_id: scryfallId, event_type: 'decrement',
    board_from: board, qty_delta: 1,
  });
}

/**
 * Move a card row from one board to another within the same deck. If the target
 * board already has a row for this card, quantities are merged; the source row
 * is deleted unconditionally. No-op when fromBoard === toBoard.
 */
export function moveCardToBoard(deckId: number, scryfallId: string, fromBoard: Board, toBoard: Board): void {
  if (fromBoard === toBoard) return;
  const db = getDb();
  db.withTransactionSync(() => {
    const src = db.getFirstSync<{ quantity: number }>(
      'SELECT quantity FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
      [deckId, scryfallId, fromBoard]
    );
    if (!src) return;
    db.runSync(
      `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board) VALUES (?, ?, ?, ?)
       ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
      [deckId, scryfallId, src.quantity, toBoard]
    );
    db.runSync(
      'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
      [deckId, scryfallId, fromBoard]
    );
    logDeckEvent({
      deck_id: deckId, scryfall_id: scryfallId, event_type: 'move',
      board_from: fromBoard, board_to: toBoard, qty_delta: src.quantity,
    });
  });
}

/**
 * Replace a card row's scryfall_id with a different printing of the same card.
 * Quantities are preserved (and merged if the target printing already has a row
 * on the same board). The target printing's `cards` row must already exist —
 * callers should `upsertCard()` the fetched printing before calling this.
 * No history event is logged: a printing change isn't an add/remove/move.
 */
export function changePrintingInDeck(
  deckId: number,
  fromScryfallId: string,
  toScryfallId: string,
  board: Board,
): void {
  if (fromScryfallId === toScryfallId) return;
  const db = getDb();
  db.withTransactionSync(() => {
    const src = db.getFirstSync<{ quantity: number }>(
      'SELECT quantity FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
      [deckId, fromScryfallId, board]
    );
    if (!src) return;
    db.runSync(
      'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
      [deckId, fromScryfallId, board]
    );
    db.runSync(
      `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board) VALUES (?, ?, ?, ?)
       ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
      [deckId, toScryfallId, src.quantity, board]
    );
  });
}

export type DeckHistoryEvent = {
  id: number;
  deck_id: number;
  scryfall_id: string;
  card_name: string;
  event_type: 'add' | 'remove' | 'decrement' | 'move';
  board_from: Board;
  board_to: Board | null;
  qty_delta: number;
  created_at: number;
};

/**
 * Reverse a single history event in-place: applies the inverse mutation to
 * deck_cards and deletes the event row. Does NOT log a new event (otherwise
 * undo would itself appear in history and could be undone, ad infinitum).
 */
export function undoDeckEvent(eventId: number): void {
  const db = getDb();
  db.withTransactionSync(() => {
    const ev = db.getFirstSync<DeckHistoryEvent>(
      `SELECT id, deck_id, scryfall_id, card_name, event_type, board_from, board_to, qty_delta, created_at
       FROM deck_history WHERE id = ?`,
      [eventId]
    );
    if (!ev) return;
    if (ev.event_type === 'add') {
      // Undo add: subtract qty_delta from board_from; drop the row if it lands at 0.
      db.runSync(
        `UPDATE deck_cards SET quantity = quantity - ?
         WHERE deck_id = ? AND scryfall_id = ? AND board = ?`,
        [ev.qty_delta, ev.deck_id, ev.scryfall_id, ev.board_from]
      );
      db.runSync(
        `DELETE FROM deck_cards
         WHERE deck_id = ? AND scryfall_id = ? AND board = ? AND quantity <= 0`,
        [ev.deck_id, ev.scryfall_id, ev.board_from]
      );
    } else if (ev.event_type === 'remove' || ev.event_type === 'decrement') {
      // Undo remove/decrement: re-add qty_delta to board_from, merging if the row already exists.
      db.runSync(
        `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board) VALUES (?, ?, ?, ?)
         ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
        [ev.deck_id, ev.scryfall_id, ev.qty_delta, ev.board_from]
      );
    } else if (ev.event_type === 'move' && ev.board_to) {
      // Undo move: pull whatever quantity is currently on board_to back to board_from.
      // If the user has since added more copies on board_to, this conservatively moves all of it.
      const src = db.getFirstSync<{ quantity: number }>(
        'SELECT quantity FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
        [ev.deck_id, ev.scryfall_id, ev.board_to]
      );
      if (src) {
        db.runSync(
          `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board) VALUES (?, ?, ?, ?)
           ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
          [ev.deck_id, ev.scryfall_id, src.quantity, ev.board_from]
        );
        db.runSync(
          'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
          [ev.deck_id, ev.scryfall_id, ev.board_to]
        );
      }
    }
    db.runSync('DELETE FROM deck_history WHERE id = ?', [eventId]);
  });
}

export function clearDeckHistory(deckId: number): void {
  getDb().runSync('DELETE FROM deck_history WHERE deck_id = ?', [deckId]);
}

export function getDeckHistory(deckId: number): DeckHistoryEvent[] {
  return getDb().getAllSync<DeckHistoryEvent>(
    `SELECT id, deck_id, scryfall_id, card_name, event_type, board_from, board_to, qty_delta, created_at
     FROM deck_history
     WHERE deck_id = ?
     ORDER BY created_at DESC, id DESC`,
    [deckId]
  );
}

export function deleteDeck(deckId: number): void {
  getDb().runSync('DELETE FROM decks WHERE id = ?', [deckId]);
}

export function exportDeckAsText(deckId: number): string {
  const cards = getDeckCards(deckId);
  const boards: Record<Board, DeckCard[]> = { main: [], side: [], commander: [], considering: [] };
  for (const card of cards) boards[card.board]?.push(card);

  const lines: string[] = [];
  if (boards.commander.length) {
    lines.push('Commander');
    boards.commander.forEach((c) => lines.push(`1 ${c.name}`));
    lines.push('');
  }
  if (boards.main.length) {
    lines.push('Deck');
    boards.main.forEach((c) => lines.push(`${c.quantity} ${c.name}`));
    lines.push('');
  }
  if (boards.side.length) {
    lines.push('Sideboard');
    boards.side.forEach((c) => lines.push(`${c.quantity} ${c.name}`));
    lines.push('');
  }
  // Considering ("maybeboard") goes last; common convention in MTGO/Arena export formats.
  if (boards.considering.length) {
    lines.push('Considering');
    boards.considering.forEach((c) => lines.push(`${c.quantity} ${c.name}`));
  }
  return lines.join('\n').trim();
}
