import { parseManaCost } from './mana';

export type CardType =
  | 'Creature' | 'Planeswalker' | 'Battle' | 'Sorcery'
  | 'Instant' | 'Artifact' | 'Enchantment' | 'Land' | 'Other';

export const CARD_TYPE_ORDER: CardType[] = [
  'Creature', 'Planeswalker', 'Battle', 'Sorcery',
  'Instant', 'Artifact', 'Enchantment', 'Land', 'Other',
];

const KNOWN: CardType[] = CARD_TYPE_ORDER.filter((t) => t !== 'Other');

export function parseTypeLine(typeLine: string): CardType {
  if (!typeLine) return 'Other';
  const head = typeLine.split(/—|-/)[0];
  for (const t of KNOWN) if (new RegExp(`\\b${t}\\b`).test(head)) return t;
  return 'Other';
}

/** Mana value: numeric tokens sum, colored / hybrid / phyrexian / colorless count as 1, X/Y/Z = 0. */
export function parseCmc(manaCost: string | null | undefined): number {
  let cmc = 0;
  for (const t of parseManaCost(manaCost)) {
    if (/^\d+$/.test(t)) cmc += parseInt(t, 10);
    else if (t === 'X' || t === 'Y' || t === 'Z') cmc += 0;
    else cmc += 1;
  }
  return cmc;
}
