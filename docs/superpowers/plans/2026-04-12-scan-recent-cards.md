# Scan Screen — Recent Cards Dropdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace auto-navigation after a successful card scan with a brief success badge + floating button that opens a slide-down panel of the last 10 scanned cards.

**Architecture:** Add `recentScans: CachedCard[]` + `addRecentScan` to the Zustand store (in-memory, not persisted). Modify `processUri` in `scan.tsx` to call `addRecentScan` and show a success badge instead of navigating. Add a floating button and animated slide-down panel to the scan screen.

**Tech Stack:** React Native, Zustand, Expo Router, `Animated` API (RN core)

---

## File Map

| File | Change |
|------|--------|
| `app/src/store/useStore.ts` | Add `recentScans`, `addRecentScan` |
| `app/__tests__/store/useStore.recentScans.test.ts` | New — tests for addRecentScan behavior |
| `app/app/(tabs)/scan.tsx` | Remove auto-nav; add success badge, floating button, slide-down panel |

---

### Task 1: Extend store with recentScans

**Files:**
- Modify: `app/src/store/useStore.ts`
- Create: `app/__tests__/store/useStore.recentScans.test.ts`

- [ ] **Step 1: Write the failing test**

Create `app/__tests__/store/useStore.recentScans.test.ts`:

```ts
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
}));

import { useStore } from '../../src/store/useStore';
import type { CachedCard } from '../../src/db/cards';

function makeCard(id: string): CachedCard {
  return {
    scryfall_id: id,
    name: `Card ${id}`,
    set_code: 'lea',
    collector_number: '1',
    mana_cost: '{1}',
    type_line: 'Instant',
    oracle_text: '',
    color_identity: '[]',
    image_uri: `https://example.com/${id}.jpg`,
    prices: JSON.stringify({ usd: '1.00' }),
    keywords: '[]',
    cached_at: Date.now(),
  };
}

describe('recentScans', () => {
  beforeEach(() => {
    useStore.setState({ recentScans: [] });
  });

  it('starts empty', () => {
    expect(useStore.getState().recentScans).toEqual([]);
  });

  it('addRecentScan prepends card (newest first)', () => {
    const { addRecentScan } = useStore.getState();
    addRecentScan(makeCard('a'));
    addRecentScan(makeCard('b'));
    const scans = useStore.getState().recentScans;
    expect(scans[0].scryfall_id).toBe('b');
    expect(scans[1].scryfall_id).toBe('a');
  });

  it('caps list at 10 entries', () => {
    const { addRecentScan } = useStore.getState();
    for (let i = 0; i < 12; i++) addRecentScan(makeCard(`card-${i}`));
    const scans = useStore.getState().recentScans;
    expect(scans).toHaveLength(10);
    expect(scans[0].scryfall_id).toBe('card-11');
    expect(scans[9].scryfall_id).toBe('card-2');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd app && npx jest --watchAll=false __tests__/store/useStore.recentScans.test.ts
```

Expected: FAIL — `recentScans` not defined on store, `addRecentScan` is not a function.

- [ ] **Step 3: Implement recentScans in store**

In `app/src/store/useStore.ts`, add the import at the top (after existing imports):

```ts
import type { CachedCard } from '../db/cards';
```

Add to the `Store` type (after the `lastScannedId` block):

```ts
  // Recent scans (in-memory, not persisted)
  recentScans: CachedCard[];
  addRecentScan: (card: CachedCard) => void;
```

Add to the `create` implementation (after the `setLastScannedId` line):

```ts
      recentScans: [],
      addRecentScan: (card) =>
        set((state) => ({
          recentScans: [card, ...state.recentScans].slice(0, 10),
        })),
```

`recentScans` is intentionally excluded from `partialize` — it already isn't listed there, so no change needed.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd app && npx jest --watchAll=false __tests__/store/useStore.recentScans.test.ts
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd app && git add src/store/useStore.ts __tests__/store/useStore.recentScans.test.ts
git commit -m "feat: add recentScans to store with addRecentScan (in-memory, max 10)"
```

---

### Task 2: Remove auto-navigation, add success badge

**Files:**
- Modify: `app/app/(tabs)/scan.tsx`

No unit tests for this task — UI behavior is tested manually.

- [ ] **Step 1: Update useStore destructure and add successCard state**

In `ScanScreen`, update the `useStore` line and add a new `useState`:

```ts
// Replace:
const { setLastScannedId } = useStore();

// With:
const { setLastScannedId, addRecentScan, recentScans } = useStore();
```

Add below the existing `useState` declarations:

```ts
const [successCard, setSuccessCard] = useState<string | null>(null);
```

- [ ] **Step 2: Replace auto-nav with addRecentScan + success badge trigger**

In `processUri`, replace this block:

```ts
      upsertCard(card);
      setLastScannedId(card.scryfall_id);
      stopScanning();
      setPickedImageUri(null);
      setPhase({ status: 'idle' });
      router.push(`/card/${card.scryfall_id}`);
```

With:

```ts
      upsertCard(card);
      addRecentScan(card);
      setLastScannedId(card.scryfall_id);
      setPhase({ status: 'idle' });
      setSuccessCard(card.name);
      setTimeout(() => {
        setSuccessCard(null);
        if (reschedule) runScanCycle();
      }, 1500);
```

Also update the `useCallback` deps comment for `processUri` — add `addRecentScan`:

```ts
  }, [stopScanning, setLastScannedId, addRecentScan]); // eslint-disable-line react-hooks/exhaustive-deps
```

(Remove `router` from deps since it's no longer used in processUri.)

- [ ] **Step 3: Add success badge to the JSX**

In the `return` block, add the success badge as a direct child of `<View style={styles.screen}>`, just after the error toast:

```tsx
      {successCard !== null && (
        <View style={scanStyles.successBadge} pointerEvents="none">
          <Text style={scanStyles.successText}>✓ {successCard}</Text>
        </View>
      )}
```

- [ ] **Step 4: Add success badge styles**

Add a new `scanStyles` StyleSheet at the bottom of the file (after `debugStyles`):

```ts
const scanStyles = StyleSheet.create({
  successBadge: {
    position: 'absolute',
    top: 56,
    alignSelf: 'center',
    zIndex: 100,
    backgroundColor: 'rgba(30,180,100,0.92)',
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 100,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 8,
  },
  successText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  // (more styles added in later tasks)
});
```

- [ ] **Step 5: Verify no TypeScript errors**

```bash
cd app && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd app && git add app/(tabs)/scan.tsx
git commit -m "feat: resume scanning after successful card scan, show success badge"
```

---

### Task 3: Add floating recent-scans button

**Files:**
- Modify: `app/app/(tabs)/scan.tsx`

- [ ] **Step 1: Add panelOpen state**

Add below the existing `useState` declarations in `ScanScreen`:

```ts
const [panelOpen, setPanelOpen] = useState(false);
```

- [ ] **Step 2: Add open/close callbacks**

Add below the existing `useRef`/`useCallback` declarations:

```ts
  const panelAnim = useRef(new Animated.Value(0)).current;

  const openPanel = useCallback(() => {
    setPanelOpen(true);
    Animated.timing(panelAnim, {
      toValue: 1,
      duration: 250,
      useNativeDriver: true,
    }).start();
  }, [panelAnim]);

  const closePanel = useCallback(() => {
    Animated.timing(panelAnim, {
      toValue: 0,
      duration: 200,
      useNativeDriver: true,
    }).start(() => setPanelOpen(false));
  }, [panelAnim]);
```

- [ ] **Step 3: Add floating button JSX**

In the `return` block, add after the success badge (and before the camera/image block):

```tsx
      {/* Floating recent-scans button */}
      <TouchableOpacity
        style={[
          scanStyles.recentBtn,
          recentScans.length === 0 && scanStyles.recentBtnDisabled,
        ]}
        onPress={panelOpen ? closePanel : openPanel}
        disabled={recentScans.length === 0}
        activeOpacity={0.75}
      >
        <Text style={scanStyles.recentBtnIcon}>⏱</Text>
        {recentScans.length > 0 && (
          <View style={scanStyles.recentBadge}>
            <Text style={scanStyles.recentBadgeText}>{recentScans.length}</Text>
          </View>
        )}
      </TouchableOpacity>
```

- [ ] **Step 4: Add floating button styles to scanStyles**

Append to the `scanStyles` StyleSheet (inside the `StyleSheet.create({...})` object):

```ts
  recentBtn: {
    position: 'absolute',
    top: 56,
    right: 16,
    zIndex: 50,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.65)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recentBtnDisabled: {
    opacity: 0.35,
  },
  recentBtnIcon: {
    fontSize: 20,
  },
  recentBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#4ecdc4',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 3,
  },
  recentBadgeText: {
    color: '#000',
    fontSize: 10,
    fontWeight: '700',
  },
```

- [ ] **Step 5: Verify no TypeScript errors**

```bash
cd app && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd app && git add app/(tabs)/scan.tsx
git commit -m "feat: add floating recent-scans button to scan screen"
```

---

### Task 4: Add slide-down recent-scans panel

**Files:**
- Modify: `app/app/(tabs)/scan.tsx`

- [ ] **Step 1: Add price helper function**

Add above the `OcrDebugPanel` component definition:

```ts
function parsePrice(pricesJson: string): string {
  try {
    const p = JSON.parse(pricesJson) as Record<string, string | undefined>;
    const val = p.usd ?? p.usd_foil;
    return val ? `$${val}` : '—';
  } catch {
    return '—';
  }
}
```

- [ ] **Step 2: Add backdrop and panel JSX**

In the `return` block, add after the floating button and before the camera/image block:

```tsx
      {/* Backdrop — closes panel on tap */}
      {panelOpen && (
        <TouchableOpacity
          style={scanStyles.backdrop}
          onPress={closePanel}
          activeOpacity={1}
        />
      )}

      {/* Slide-down recent scans panel */}
      <Animated.View
        style={[
          scanStyles.panel,
          {
            transform: [
              {
                translateY: panelAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [-320, 0],
                }),
              },
            ],
          },
        ]}
        pointerEvents={panelOpen ? 'auto' : 'none'}
      >
        <FlatList
          data={recentScans}
          keyExtractor={(item) => item.scryfall_id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={scanStyles.row}
              onPress={() => { closePanel(); stopScanning(); router.push(`/card/${item.scryfall_id}`); }}
              activeOpacity={0.7}
            >
              <Image
                source={{ uri: item.image_uri }}
                style={scanStyles.thumb}
                resizeMode="cover"
              />
              <View style={scanStyles.rowInfo}>
                <Text style={scanStyles.rowName} numberOfLines={1}>{item.name}</Text>
                <Text style={scanStyles.rowSet}>
                  {item.set_code.toUpperCase()} · #{item.collector_number}
                </Text>
              </View>
              <Text style={scanStyles.rowPrice}>{parsePrice(item.prices)}</Text>
            </TouchableOpacity>
          )}
          ItemSeparatorComponent={() => <View style={scanStyles.separator} />}
        />
      </Animated.View>
```

- [ ] **Step 3: Add panel and row styles to scanStyles**

Append to the `scanStyles` StyleSheet:

```ts
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 48,
  },
  panel: {
    position: 'absolute',
    top: 108,
    left: 12,
    right: 12,
    maxHeight: 320,
    zIndex: 49,
    backgroundColor: 'rgba(0,0,0,0.88)',
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.12)',
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 12,
  },
  thumb: {
    width: 48,
    height: 68,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  rowInfo: {
    flex: 1,
    gap: 3,
  },
  rowName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  rowSet: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 11,
  },
  rowPrice: {
    color: '#4ecdc4',
    fontSize: 13,
    fontWeight: '500',
    minWidth: 44,
    textAlign: 'right',
  },
  separator: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(255,255,255,0.1)',
    marginLeft: 72,
  },
```

- [ ] **Step 4: Add FlatList to imports**

At the top of `scan.tsx`, `FlatList` is not yet imported. Add it to the existing RN import:

```ts
// Replace:
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ScrollView,
} from 'react-native';

// With:
import {
  View,
  Text,
  Image,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ScrollView,
} from 'react-native';
```

- [ ] **Step 5: Verify no TypeScript errors**

```bash
cd app && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd app && git add app/(tabs)/scan.tsx
git commit -m "feat: add recent-scans slide-down panel to scan screen"
```
