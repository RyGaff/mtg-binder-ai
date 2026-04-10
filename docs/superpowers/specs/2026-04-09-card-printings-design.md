# Card Printings Section Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## Goal

Show all printings of a card below the Similar Cards section on the card detail screen, with foil and non-foil prices for each printing, sorted newest first.

---

## Layout

A `CardPrintings` component sits directly below `FindSimilar` on the card detail screen. It uses a vertical list (not horizontal scroll) so users can compare prices across rows.

Each row displays:
- **Set square**: small colored square (`#4ecdc4` teal) with the 3-4 letter set code in white
- **Set name**: left-aligned, truncated if too long
- **Collector number**: muted, displayed after set name
- **Non-foil price**: right-aligned (USD from `prices.usd`), `—` if unavailable
- **Foil price**: right-aligned with `✦` prefix (from `prices.usd_foil`), `—` if unavailable
- Tapping a row navigates to that printing's card detail page (`/card/<scryfall_id>`)

The section is hidden entirely if the card has only one printing (or zero results).

Loading and error states match the `FindSimilar` pattern (ActivityIndicator / muted error text).

---

## Data & API

Scryfall printings endpoint:
```
GET /cards/search?q=!"Card Name"&unique=prints&order=released&dir=desc
```

Returns all printings of a card sorted newest first. Results are not stored in SQLite — prices change daily so no caching.

### New type

```typescript
type PrintingSummary = {
  scryfall_id: string;
  set_code: string;       // e.g. "IKO"
  set_name: string;       // e.g. "Ikoria: Lair of Behemoths"
  collector_number: string;
  prices: {
    usd: string | null;
    usd_foil: string | null;
  };
};
```

### New API function

`fetchPrintings(name: string): Promise<PrintingSummary[]>` in `app/src/api/scryfall.ts`

Maps Scryfall response fields: `id → scryfall_id`, `set → set_code`, `set_name`, `collector_number`, `prices`.

### New hook

`usePrintings(card: CachedCard)` in `app/src/api/hooks.ts`

- Uses TanStack Query with `queryKey: ['printings', card.name]`
- `staleTime: 30 * 60 * 1000` (30 min — prices are semi-fresh)
- Enabled immediately (no embedding dependency)

---

## Files Affected

| File | Change |
|------|--------|
| `app/src/api/scryfall.ts` | Add `fetchPrintings(name)` |
| `app/src/api/hooks.ts` | Add `usePrintings(card)` hook |
| `app/src/components/CardPrintings.tsx` | New component |
| `app/app/card/[id].tsx` | Add `<CardPrintings card={card} />` below `<FindSimilar />` |

---

## Error Handling

- Network error → show muted "Could not load printings" text (same pattern as FindSimilar)
- Single printing → hide section entirely (don't show a list of one)
- Missing price → display `—`
