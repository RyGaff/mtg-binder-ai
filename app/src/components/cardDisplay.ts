export type CardDisplay =
  | { kind: 'normal' }
  | { kind: 'two-image' }                                           // tap flips to uriBack; text swaps to face[1]
  | { kind: 'rotate'; degrees: number; textMode: 'stack' | 'swap' } // tap rotates; text stacks (split) or swaps (flip)
  | { kind: 'split-text' };                                         // shared image, both face texts stacked (adventure)

export function cardDisplay(layout: string | undefined, faceCount: number): CardDisplay {
  switch (layout) {
    case 'transform':
    case 'modal_dfc':
    case 'double_faced_token':
    case 'reversible_card':
    case 'art_series':
    case 'meld':
      return { kind: 'two-image' };
    case 'split':
      return { kind: 'rotate', degrees: 90, textMode: 'stack' };
    case 'flip':
      return { kind: 'rotate', degrees: 180, textMode: 'swap' };
    case 'adventure':
      return { kind: 'split-text' };
    default:
      // Fallback for rows cached before the `layout` column existed.
      return faceCount >= 2 ? { kind: 'two-image' } : { kind: 'normal' };
  }
}

export type CardLike = {
  image_uri?: string;
  image_uri_back?: string;
  layout?: string;
  card_faces?: string; // JSON string
};

function parseFaceCount(raw: string | undefined): number {
  if (!raw) return 0;
  try {
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.length : 0;
  } catch {
    return 0;
  }
}

export function cardImageTransform(card: CardLike): { uri: string; uriBack?: string; rotateDeg?: number } {
  const display = cardDisplay(card.layout, parseFaceCount(card.card_faces));
  const uri = card.image_uri ?? '';
  switch (display.kind) {
    case 'two-image':
      return { uri, uriBack: card.image_uri_back || undefined };
    case 'rotate':
      return { uri, rotateDeg: display.degrees };
    default:
      return { uri };
  }
}
