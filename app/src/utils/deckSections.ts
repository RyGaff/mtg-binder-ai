import type { DeckCard } from '../db/decks';
import { CARD_TYPE_ORDER, parseCmc, parseTypeLine } from './cardHelpers';
import { boardPrice } from './deckStats';

export type RowSection = {
  /** Bold board-level label OR type sub-label inside Main. */
  title: string;
  /** "board" headers carry count + price; "type" headers carry just count. */
  kind: 'board' | 'type';
  count: number;
  price?: number;
  data: DeckCard[];
};

/**
 * Output order:
 *   Commander (board, flat list)
 *   Main (board header) → Creatures / Planeswalkers / ... / Lands (type sub-sections)
 *   Sideboard (board, flat list)
 *   Considering (board, flat list — "maybeboard" / cards on the bubble)
 */
export function buildSections(cards: DeckCard[]): RowSection[] {
  const commander = cards.filter((c) => c.board === 'commander');
  const main = cards.filter((c) => c.board === 'main');
  const side = cards.filter((c) => c.board === 'side');
  const considering = cards.filter((c) => c.board === 'considering');

  const sumQty = (xs: DeckCard[]) => xs.reduce((s, c) => s + c.quantity, 0);
  const sortInner = (a: DeckCard, b: DeckCard) =>
    parseCmc(a.mana_cost ?? '') - parseCmc(b.mana_cost ?? '') || a.name.localeCompare(b.name);

  const out: RowSection[] = [];

  if (commander.length > 0) {
    out.push({ title: 'Commander', kind: 'board', count: sumQty(commander), price: boardPrice(commander), data: [...commander].sort(sortInner) });
  }

  if (main.length > 0) {
    out.push({ title: 'Main', kind: 'board', count: sumQty(main), price: boardPrice(main), data: [] });
    const byType: Record<string, DeckCard[]> = {};
    for (const c of main) {
      const t = parseTypeLine(c.type_line ?? '');
      (byType[t] ??= []).push(c);
    }
    for (const t of CARD_TYPE_ORDER) {
      if (!byType[t]?.length) continue;
      out.push({ title: t, kind: 'type', count: sumQty(byType[t]), data: [...byType[t]].sort(sortInner) });
    }
  }

  if (side.length > 0) {
    out.push({ title: 'Sideboard', kind: 'board', count: sumQty(side), price: boardPrice(side), data: [...side].sort(sortInner) });
  }

  if (considering.length > 0) {
    out.push({ title: 'Considering', kind: 'board', count: sumQty(considering), price: boardPrice(considering), data: [...considering].sort(sortInner) });
  }

  return out;
}
