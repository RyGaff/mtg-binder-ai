import { getDb } from './db';
import type { CachedCard } from './cards';

type Board = 'main' | 'side' | 'commander';

export type Deck = {
  id: number;
  name: string;
  format: string;
  created_at: number;
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
  return getDb().getAllSync<Deck>('SELECT id, name, format, created_at FROM decks ORDER BY created_at DESC');
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

export function deleteDeck(deckId: number): void {
  getDb().runSync('DELETE FROM decks WHERE id = ?', [deckId]);
}

export function exportDeckAsText(deckId: number): string {
  const cards = getDeckCards(deckId);
  const boards: Record<Board, DeckCard[]> = { main: [], side: [], commander: [] };
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
  }
  return lines.join('\n').trim();
}
