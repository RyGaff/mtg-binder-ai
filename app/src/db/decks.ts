import { getDb } from './db';
import type { CachedCard } from './cards';

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
  board: 'main' | 'side' | 'commander';
} & CachedCard;

export type AddCardToDeckArgs = {
  deck_id: number;
  scryfall_id: string;
  quantity: number;
  board: 'main' | 'side' | 'commander';
};

export function createDeck(args: { name: string; format: string }): number {
  const db = getDb();
  const result = db.runSync(
    'INSERT INTO decks (name, format, created_at) VALUES (?, ?, ?)',
    [args.name, args.format, Date.now()]
  );
  return result.lastInsertRowId;
}

export function getDecks(): Deck[] {
  const db = getDb();
  return db.getAllSync<Deck>('SELECT id, name, format, created_at FROM decks ORDER BY created_at DESC');
}

export function getDeckCards(deckId: number): DeckCard[] {
  const db = getDb();
  return db.getAllSync<DeckCard>(
    `SELECT dc.deck_id, dc.scryfall_id, dc.quantity, dc.board,
            c.name, c.set_code, c.collector_number, c.mana_cost,
            c.type_line, c.oracle_text, c.color_identity, c.image_uri,
            c.prices, c.keywords, c.cached_at
     FROM deck_cards dc
     JOIN cards c ON c.scryfall_id = dc.scryfall_id
     WHERE dc.deck_id = ?
     ORDER BY dc.board, c.mana_cost, c.name`,
    [deckId]
  );
}

export function addCardToDeck(args: AddCardToDeckArgs): void {
  const db = getDb();
  db.runSync(
    `INSERT INTO deck_cards (deck_id, scryfall_id, quantity, board)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(deck_id, scryfall_id, board) DO UPDATE SET quantity = excluded.quantity`,
    [args.deck_id, args.scryfall_id, args.quantity, args.board]
  );
}

export function removeCardFromDeck(deckId: number, scryfallId: string, board: 'main' | 'side' | 'commander'): void {
  const db = getDb();
  db.runSync(
    'DELETE FROM deck_cards WHERE deck_id = ? AND scryfall_id = ? AND board = ?',
    [deckId, scryfallId, board]
  );
}

export function deleteDeck(deckId: number): void {
  const db = getDb();
  db.runSync('DELETE FROM decks WHERE id = ?', [deckId]);
}

export function exportDeckAsText(deckId: number): string {
  const cards = getDeckCards(deckId);
  const boards: Record<string, DeckCard[]> = { main: [], side: [], commander: [] };
  for (const card of cards) boards[card.board]?.push(card);

  const lines: string[] = [];
  if (boards.commander.length) {
    lines.push('Commander');
    boards.commander.forEach(c => lines.push(`1 ${c.name}`));
    lines.push('');
  }
  if (boards.main.length) {
    lines.push('Deck');
    boards.main.forEach(c => lines.push(`${c.quantity} ${c.name}`));
    lines.push('');
  }
  if (boards.side.length) {
    lines.push('Sideboard');
    boards.side.forEach(c => lines.push(`${c.quantity} ${c.name}`));
  }
  return lines.join('\n').trim();
}
