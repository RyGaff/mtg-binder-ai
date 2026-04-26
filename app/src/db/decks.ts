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
  color_identity: string[];
};

type DeckMetaRow = {
  id: number;
  name: string;
  format: string;
  created_at: number;
  art_crop_uri: string;
  card_count: number | null;
  color_identity_concat: string | null;
};

export function getDecksWithMeta(): DeckWithMeta[] {
  const rows = getDb().getAllSync<DeckMetaRow>(`
    SELECT
      d.id, d.name, d.format, d.created_at, d.art_crop_uri,
      COALESCE(SUM(dc.quantity), 0) AS card_count,
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

export function addCardToDeck(args: AddCardToDeckArgs): void {
  getDb().runSync(
    `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = quantity + excluded.quantity`,
    [args.deck_id, args.scryfall_id, args.quantity, args.board]
  );
}

export function removeCardFromDeck(deckId: number, scryfallId: string, board: Board): void {
  getDb().runSync(
    'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
    [deckId, scryfallId, board]
  );
}

/** Decrement qty by 1; deletes the row if qty drops to 0. */
export function decrementCardInDeck(deckId: number, scryfallId: string, board: Board): void {
  const db = getDb();
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
