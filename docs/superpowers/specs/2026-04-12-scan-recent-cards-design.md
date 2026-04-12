# Scan Screen — Recent Cards Dropdown

**Date:** 2026-04-12

## Overview

Replace the current behavior of auto-navigating to the card detail screen after a successful scan. Instead, scanning resumes automatically and a floating button lets users open a slide-down panel of recently scanned cards.

---

## 1. Data Layer

### Store changes (`app/src/store/useStore.ts`)

Add two fields to the `Store` type and implementation:

```ts
recentScans: CachedCard[];
addRecentScan: (card: CachedCard) => void;
```

- `recentScans` starts as `[]`, max length 10, newest first.
- `addRecentScan` prepends the card and slices to 10: `[card, ...prev].slice(0, 10)`.
- **Not persisted** — excluded from `partialize`. Resets on app restart.
- Import `CachedCard` from `../../db/cards` in the store file.

---

## 2. Post-Scan Behavior

In `processUri` (`scan.tsx`), replace the current block:

```ts
setLastScannedId(card.scryfall_id);
stopScanning();
setPickedImageUri(null);
setPhase({ status: 'idle' });
router.push(`/card/${card.scryfall_id}`);
```

With:

```ts
addRecentScan(card);
setLastScannedId(card.scryfall_id);
setPhase({ status: 'idle' });
// Show success badge for 1.5s, then resume scanning
setSuccessCard(card.name);   // local state, drives the badge
setTimeout(() => {
  setSuccessCard(null);
  if (reschedule) runScanCycle();  // reschedule=false for photo-library flow
}, 1500);
```

For picked-image flow (non-live camera), scanning does not resume — just show the badge.

**Success badge:** Absolute-positioned pill near the top of the camera view. Text: `"✓ {card.name}"`. Fades in, stays 1.5s, removed when `successCard` returns to null (no fade-out animation needed).

---

## 3. Floating Button

Position: `position: 'absolute'`, top-right of the camera/image view, `top: 12`, `right: 12`, `zIndex: 50`.

- Always rendered but disabled (reduced opacity) when `recentScans.length === 0`.
- Shows a small count badge (e.g. `"3"`) over the button when `recentScans.length > 0`.
- Icon: a simple clock/history icon or the text `"Recent"` — use a list icon character or label, no external icon library needed.
- Tap: toggles `panelOpen` local state → drives slide animation.

---

## 4. Slide-Down Panel

### Animation

```ts
const panelAnim = useRef(new Animated.Value(0)).current;
// 0 = hidden (translateY: -panelHeight), 1 = visible (translateY: 0)
```

`Animated.timing` with `duration: 250`, `useNativeDriver: true`.

Panel height: fixed at `~320px` (fits ~4–5 rows comfortably, scrollable for more).

### Layout

- Anchored `position: 'absolute'` below the floating button, `top: ~52`, `right: 12`, `left: 12`, `zIndex: 49`.
- Semi-transparent dark background (`rgba(0,0,0,0.88)`), `borderRadius: 12`, matching existing debug panel aesthetic.
- `FlatList` of `recentScans`, `keyExtractor` by `scryfall_id`.

### Row layout

Each row is a `TouchableOpacity`:

```
[ thumbnail 48×68 ] [ name (bold)          ] [ $price ]
                    [ SET · #num            ]
```

- Thumbnail: `Image` from `card.image_uri`, `width: 48`, `height: 68`, `borderRadius: 4`.
- Name: `fontSize: 14`, `fontWeight: '600'`, `color: theme.text`.
- Set line: `fontSize: 11`, `color: theme.textSecondary`, formatted as `SET.toUpperCase() · #collector_number`.
- Price: parse `JSON.parse(card.prices)`, prefer `usd`, fallback `usd_foil`, fallback `"—"`. Display as `$X.XX` right-aligned, `fontSize: 13`, `color: theme.accent`.
- Tap: `stopScanning()` + `router.push('/card/${card.scryfall_id}')`.

### Backdrop

A full-screen transparent `TouchableOpacity` rendered behind the panel when open. Tap closes the panel. `zIndex: 48`.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `app/src/store/useStore.ts` | Add `recentScans`, `addRecentScan` |
| `app/app/(tabs)/scan.tsx` | Remove nav, add success badge, floating button, slide-down panel |

No new files required.

---

## 6. Out of Scope

- Persisting recent scans across app restarts
- Clearing the recent scans list manually
- Showing foil indicator in the panel row
