import { ActivityIndicator, FlatList, Pressable, RefreshControl, StyleSheet, Text, TouchableOpacity, useWindowDimensions, View } from 'react-native';
import { Image } from 'expo-image';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import {
  addCardToDeck, clearDeckHistory, decrementCardInDeck, deleteDeck, exportDeckAsText,
  getDeck, getDeckCards, moveCardToBoard, removeCardFromDeck, setDeckArt,
  type Board, type DeckCard,
} from '../../src/db/decks';
import { fetchArtCrop } from '../../src/api/scryfall';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';
import { AddCardsSheet } from '../../src/components/AddCardsSheet';
import { ChangePrintingSheet } from '../../src/components/ChangePrintingSheet';
import { useActionSheet } from '../../src/components/ActionSheet';
import { DeckHero } from '../../src/components/DeckHero';
import { DeckInfoStrip } from '../../src/components/DeckInfoStrip';
import { DeckStatsPanel } from '../../src/components/DeckStatsPanel';
import { DeckHistoryPanel } from '../../src/components/DeckHistoryPanel';
import { ManaCost } from '../../src/components/ManaCost';
import { boardPrice, cardPriceUsd } from '../../src/utils/deckStats';
import { buildSections, type RowSection } from '../../src/utils/deckSections';

export default function DeckDetailScreen() {
  const t = useTheme();
  const router = useRouter();
  const qc = useQueryClient();
  const { id } = useLocalSearchParams<{ id: string }>();
  const deckId = Number(id);
  const setActiveDeckId = useStore((s) => s.setActiveDeckId);
  // Per-deck view mode lives in client state (Zustand, persisted) — switching is
  // a UI preference, not a property of the deck, so it never touches the DB.
  const viewMode = useStore((s) => s.deckViewModes[deckId] ?? 'list');
  const setDeckViewMode = useStore((s) => s.setDeckViewMode);
  const insets = useSafeAreaInsets();

  // Deck row + its cards. TanStack Query keeps both fresh across navigations.
  // `enabled` gates the queries on a finite numeric id so a bad route param doesn't fire bogus reads.
  const deckQ = useQuery({ queryKey: ['deck', deckId], queryFn: () => getDeck(deckId), enabled: Number.isFinite(deckId) });
  const cardsQ = useQuery({ queryKey: ['deck-cards', deckId], queryFn: () => getDeckCards(deckId), enabled: Number.isFinite(deckId) });
  const deck = deckQ.data;
  const cards = cardsQ.data ?? [];
  const isLoading = deckQ.isPending || cardsQ.isPending;
  const isError = deckQ.isError || cardsQ.isError;
  const deckMissing = !isLoading && !isError && !deck;
  const refreshing = deckQ.isFetching || cardsQ.isFetching;
  const onRefresh = useCallback(() => { deckQ.refetch(); cardsQ.refetch(); }, [deckQ, cardsQ]);

  // Local UI state: stats panel collapse, add-card sheet visibility, generic action sheet hook.
  // Stats and History panels are mutually exclusive — opening one closes the other so
  // the page doesn't grow two large stacked panels.
  const [statsOpen, setStatsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  // Card whose printing the user is currently swapping. Non-null = sheet visible.
  const [printingTarget, setPrintingTarget] = useState<DeckCard | null>(null);
  // Speed-dial open state. When true the FAB cluster expands: an "Add card"
  // button stacks above the FAB and a column of nav buttons (Top / Main /
  // Sideboard / Considering) sits to the left.
  const [fabOpen, setFabOpen] = useState(false);
  // FlatList ref drives the nav-button scroll targets. FlatRow is declared
  // further down in this component, so the ref is left ungeneric.
  const listRef = useRef<FlatList>(null);
  // Live scroll offset, kept in a ref so onScroll updates don't re-render the
  // screen. Read only when a nav button captures the user's "previous spot"
  // before teleporting elsewhere.
  const currentOffsetRef = useRef(0);
  // Saved offset from the last nav-button press. null = nothing to restore yet.
  // State (not ref) because the Restore button's disabled prop reflects it.
  const [savedOffset, setSavedOffset] = useState<number | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  // Set when the user inspects a card from inside AddCardsSheet — read on next focus
  // to auto-reopen the sheet so its retained search reappears.
  const resumeAddOnFocus = useRef(false);
  // Considering ("maybeboard") collapses by default — those cards aren't in the deck
  // proper, so they shouldn't crowd the scroll surface unless the user opts in.
  const [consideringOpen, setConsideringOpen] = useState(false);
  const sheet = useActionSheet();

  // Grid view geometry — tile target ~120pt wide; columns derived from window width
  // so phones land on 2–3 cols, larger tablets get more. Tile aspect locked to MTG 5:7.
  const { width: screenWidth } = useWindowDimensions();
  const GRID_PADDING = 14;
  const GRID_GAP = 8;
  const TILE_TARGET = 120;
  const gridCols = Math.max(2, Math.floor((screenWidth - GRID_PADDING * 2 + GRID_GAP) / (TILE_TARGET + GRID_GAP)));
  const tileWidth = Math.floor((screenWidth - GRID_PADDING * 2 - GRID_GAP * (gridCols - 1)) / gridCols);
  const tileHeight = Math.round(tileWidth * 7 / 5);

  // Derived view-model values, all memoized on `cards`.
  const sections = useMemo(() => buildSections(cards), [cards]);
  const mainCount = useMemo(() => cards.filter((c) => c.board === 'main').reduce((s, c) => s + c.quantity, 0), [cards]);
  const sideCount = useMemo(() => cards.filter((c) => c.board === 'side').reduce((s, c) => s + c.quantity, 0), [cards]);
  const totalPrice = useMemo(() => boardPrice(cards), [cards]);
  // Aggregate WUBRG color identity across every card in the deck.
  const colorIdentity = useMemo(() => {
    const set = new Set<string>();
    for (const c of cards) {
      try { for (const k of JSON.parse(c.color_identity || '[]') as string[]) if ('WUBRG'.includes(k)) set.add(k); }
      catch { /* skip */ }
    }
    return Array.from(set);
  }, [cards]);

  // Stats panel inputs — split by board so the panel can decide whether to fold the
  // commander into the "main" price (Commander-format only). Curve/colors/types still
  // combine main + commander internally.
  const mainCards = useMemo(() => cards.filter((c) => c.board === 'main'), [cards]);
  const commanderCards = useMemo(() => cards.filter((c) => c.board === 'commander'), [cards]);
  const sideCards = useMemo(() => cards.filter((c) => c.board === 'side'), [cards]);
  // Commander color identity — merged across all cards on the commander board
  // (covers partner/background pairs). Lowercased + ordered WUBRG; empty CI
  // collapses to 'c' so Scryfall's `commander:c` returns colorless legals.
  const commanderColorIdentity = useMemo(() => {
    const set = new Set<string>();
    for (const c of commanderCards) {
      try {
        for (const k of JSON.parse(c.color_identity || '[]') as string[]) {
          const lk = k.toLowerCase();
          if ('wubrg'.includes(lk)) set.add(lk);
        }
      } catch { /* skip malformed JSON */ }
    }
    if (set.size === 0) return 'c';
    return ['w', 'u', 'b', 'r', 'g'].filter((c) => set.has(c)).join('');
  }, [commanderCards]);

  // Long-press on any card row → action sheet with Set as deck art / Remove options.
  const cardOptions = useCallback((card: DeckCard) => {
    // Shared post-mutation invalidations: refresh card list, deck row (card_count), deck index, history.
    const invalidateAll = () => {
      qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
      qc.invalidateQueries({ queryKey: ['deck', deckId] });
      qc.invalidateQueries({ queryKey: ['decks'] });
      qc.invalidateQueries({ queryKey: ['deck-history', deckId] });
    };
    // "Remove all" is a destructive irreversible op — chain into a confirm sheet.
    const confirmRemoveAll = () => {
      sheet.show({
        title: `Remove all copies of ${card.name}?`,
        actions: [
          { label: 'Remove all', destructive: true, onPress: () => {
            removeCardFromDeck(deckId, card.scryfall_id, card.board);
            invalidateAll();
          } },
        ],
      });
    };
    const changePrintingAction = { label: 'Change printing', onPress: () => setPrintingTarget(card) };
    const setArtAction = { label: 'Set as deck art', onPress: async () => {
      try {
        const uri = await fetchArtCrop(card.scryfall_id);
        if (uri) {
          setDeckArt(deckId, uri);
          qc.invalidateQueries({ queryKey: ['deck', deckId] });
          qc.invalidateQueries({ queryKey: ['decks'] });
        }
      } catch (e) { console.warn('fetchArtCrop failed', e); }
    } };
    // "Move to <board>" — one entry per board the card isn't already on. Quantities
    // merge if a row already exists on the target. Order mirrors the section order
    // on screen (Commander → Main → Sideboard → Considering).
    const BOARD_LABEL: Record<Board, string> = {
      commander: 'Commander', main: 'Main', side: 'Sideboard', considering: 'Considering',
    };
    const ALL_BOARDS: Board[] = ['commander', 'main', 'side', 'considering'];
    const moveActions = ALL_BOARDS
      .filter((b) => b !== card.board)
      .map((b) => ({ label: `Move to ${BOARD_LABEL[b]}`, onPress: () => {
        moveCardToBoard(deckId, card.scryfall_id, card.board, b);
        invalidateAll();
      } }));
    // Single-copy: only show one destructive Remove (no need for the qty distinction).
    if (card.quantity <= 1) {
      sheet.show({
        title: card.name,
        actions: [
          setArtAction,
          changePrintingAction,
          ...moveActions,
          { label: 'Remove', destructive: true, onPress: () => {
            removeCardFromDeck(deckId, card.scryfall_id, card.board);
            invalidateAll();
          } },
        ],
      });
      return;
    }
    sheet.show({
      title: card.name,
      actions: [
        setArtAction,
        ...moveActions,
        { label: 'Remove 1', onPress: () => {
          decrementCardInDeck(deckId, card.scryfall_id, card.board);
          invalidateAll();
        } },
        { label: 'Remove all', destructive: true, onPress: confirmRemoveAll },
      ],
    });
  }, [deckId, qc, sheet]);

  // Serialize deck → text file → native Share sheet.
  const exportDeck = useCallback(async () => {
    try {
      const text = exportDeckAsText(deckId);
      // Sanitize: collapse anything outside [\w.-] to underscore, cap length, fall back to 'deck'.
      const safeName = (deck?.name ?? 'deck').replace(/[^\w.-]+/g, '_').slice(0, 60) || 'deck';
      const path = `${FileSystem.cacheDirectory}${safeName}.txt`;
      await FileSystem.writeAsStringAsync(path, text, { encoding: FileSystem.EncodingType.UTF8 });
      await Sharing.shareAsync(path, { mimeType: 'text/plain' });
    } catch (e) { console.warn('exportDeck failed', e); }
  }, [deckId, deck?.name]);

  // Hero ⋮ → deck-level action sheet: Export + Delete deck (with confirm).
  const more = useCallback(() => {
    const name = deck?.name ?? 'Deck';
    // Two-step confirm so Delete isn't a single mis-tap away.
    const confirmDelete = () => {
      sheet.show({
        title: `Delete "${name}"?`,
        actions: [
          { label: 'Delete', destructive: true, onPress: () => {
            deleteDeck(deckId);
            qc.invalidateQueries({ queryKey: ['decks'] });
            router.back();
          } },
        ],
      });
    };
    const toggleView = () => {
      setDeckViewMode(deckId, viewMode === 'grid' ? 'list' : 'grid');
    };
    // "Clear history" is destructive (it can't be undone — the deck_history rows
    // are gone), so chain into a confirm sheet rather than wiping on first tap.
    const confirmClearHistory = () => {
      sheet.show({
        title: `Clear history for "${name}"?`,
        subtitle: 'Past add/remove/move events will be deleted. Cards in the deck are unaffected.',
        actions: [
          { label: 'Clear history', destructive: true, onPress: () => {
            clearDeckHistory(deckId);
            qc.invalidateQueries({ queryKey: ['deck-history', deckId] });
          } },
        ],
      });
    };
    sheet.show({
      title: name,
      actions: [
        { label: viewMode === 'grid' ? 'View: List' : 'View: Grid', onPress: toggleView },
        { label: 'Export', onPress: () => { void exportDeck(); } },
        { label: 'Clear history', onPress: confirmClearHistory },
        { label: 'Delete deck', destructive: true, onPress: confirmDelete },
      ],
    });
  }, [deck?.name, deckId, exportDeck, qc, router, sheet, viewMode, setDeckViewMode]);

  // FAB → opens in-page AddCardsSheet. Sets active deck so card-detail "Add to deck" knows the target.
  const openAdd = useCallback(() => {
    setActiveDeckId(deckId);
    setAddOpen(true);
    setFabOpen(false);
  }, [deckId, setActiveDeckId]);


  // Re-focus handler — fires whenever this screen regains focus (mount, return from
  // child route). If the user just left to inspect a card from inside the sheet, the
  // ref flag is true; reopen the sheet so its retained search snaps back into view.
  // Deferred by a frame so the navigator's modal dismissal animation finishes
  // committing before we mount the RN Modal again — otherwise iOS can land in a
  // state where a subsequent close leaves an invisible touch interceptor.
  useFocusEffect(
    useCallback(() => {
      if (resumeAddOnFocus.current) {
        resumeAddOnFocus.current = false;
        const id = requestAnimationFrame(() => setAddOpen(true));
        return () => cancelAnimationFrame(id);
      }
    }, [])
  );

  // Section header — bold for boards (Commander/Main/Sideboard), small caps for type sub-sections.
  // Returns plain JSX (not a FlatList row renderer) so we can compose it inside the flattened
  // row dispatcher below without paying SectionList's sticky-header iOS quirk.
  // The Considering board renders as a Pressable with a chevron — tap toggles its rows
  // in/out of the flattened list. Other headers stay non-interactive.
  const renderSectionHeader = useCallback((section: RowSection) => {
    const isBoard = section.kind === 'board';
    const isConsidering = section.title === 'Considering';
    const countText = `${section.count} ${section.count === 1 ? 'card' : 'cards'}${section.price != null ? ` · $${section.price.toFixed(2)}` : ''}`;
    if (isConsidering) {
      return (
        <Pressable
          onPress={() => setConsideringOpen((v) => !v)}
          style={({ pressed }) => [s.section, s.boardSection, { borderTopColor: t.border }, pressed && { opacity: 0.6 }]}
        >
          <Text style={[s.boardTitle, { color: t.text }]}>
            {consideringOpen ? '▾ ' : '▸ '}{section.title}
          </Text>
          <Text style={[s.sectionCount, { color: t.textSecondary }]}>{countText}</Text>
        </Pressable>
      );
    }
    return (
      <View style={[s.section, isBoard ? s.boardSection : null, isBoard ? { borderTopColor: t.border } : null]}>
        <Text style={[isBoard ? s.boardTitle : s.typeTitle, { color: isBoard ? t.text : t.textSecondary }]}>
          {section.title}
        </Text>
        <Text style={[s.sectionCount, { color: t.textSecondary }]}>{countText}</Text>
      </View>
    );
  }, [t, consideringOpen]);

  // Shared invalidation for any qty mutation on a row's card.
  const invalidateAfterRowMutation = useCallback(() => {
    qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
    qc.invalidateQueries({ queryKey: ['deck', deckId] });
    qc.invalidateQueries({ queryKey: ['decks'] });
    qc.invalidateQueries({ queryKey: ['deck-history', deckId] });
  }, [qc, deckId]);

  // Inline + / − stepper handlers. + adds another copy; − decrements (delete on 0).
  // Stepper interactions also close the history panel — once the user starts
  // editing a row, the timeline above is stale until the new event lands and
  // forcing them to scroll past it adds noise.
  const incCard = useCallback((item: DeckCard) => {
    addCardToDeck({ deck_id: deckId, scryfall_id: item.scryfall_id, quantity: 1, board: item.board });
    invalidateAfterRowMutation();
    setHistoryOpen(false);
  }, [deckId, invalidateAfterRowMutation]);
  const decCard = useCallback((item: DeckCard) => {
    decrementCardInDeck(deckId, item.scryfall_id, item.board);
    invalidateAfterRowMutation();
    setHistoryOpen(false);
  }, [deckId, invalidateAfterRowMutation]);

  // Shared text-row builder — [−][qty][+] · mana · name · price · ⋮. The leftmost
  // stepper is the primary qty editor; ⋮ remains for "Set as deck art" / "Remove
  // all". Stepper buttons are siblings of the main pressable so their taps don't
  // navigate the user to /card/[id].
  const renderTextRow = useCallback((item: DeckCard) => {
    const price = cardPriceUsd(item);
    return (
      <View style={[s.row, { borderBottomColor: t.border }]}>
        <TouchableOpacity
          onPress={() => cardOptions(item)}
          hitSlop={8}
          accessibilityRole="button"
          accessibilityLabel={`Options for ${item.name}`}
          style={s.rowMenu}
        >
          <Text style={[s.rowMenuIcon, { color: t.textSecondary }]}>⋮</Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => router.push(`/card/${item.scryfall_id}`)}
          onLongPress={() => cardOptions(item)}
          delayLongPress={400}
          style={s.rowMain}
        >
          <ManaCost cost={item.mana_cost} size={12} />
          <Text style={[s.name, { color: t.text }]} numberOfLines={2}>{item.name}</Text>
          <Text style={[s.price, { color: t.textSecondary }]}>
            {price != null ? `$${price.toFixed(2)}` : '—'}
          </Text>
        </TouchableOpacity>
        <View style={s.stepper}>
          <TouchableOpacity
            onPress={() => decCard(item)}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel={`Remove one ${item.name}`}
            style={[s.stepBtn, { borderColor: t.border }]}
          >
            <Text style={[s.stepIcon, { color: t.textSecondary }]}>−</Text>
          </TouchableOpacity>
          <Text style={[s.stepCount, { color: t.text }]}>{item.quantity}</Text>
          <TouchableOpacity
            onPress={() => incCard(item)}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel={`Add one ${item.name}`}
            style={[s.stepBtn, { borderColor: t.border }]}
          >
            <Text style={[s.stepIcon, { color: t.textSecondary }]}>+</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }, [router, cardOptions, t, incCard, decCard]);

  // Commander section row — stacked image tile (220pt wide, MTG 5:7 aspect) with name +
  // mana cost beneath. Three things keep vertical scroll alive over this row:
  //   (1) Image has pointerEvents="none" so it can never become the touch responder.
  //   (2) The label Pressable sizes to its text content (no minWidth) and uses a short
  //       delayLongPress so scroll-vs-press handoff resolves quickly.
  //   (3) The label Text uses `commanderName` (no flex:1) instead of `s.name`; flex:1
  //       on a Text inside a column-flex container makes yoga grow it unboundedly,
  //       which corrupts the row height FlatList relies on to compute scroll offsets.
  const renderCommanderItem = useCallback((item: DeckCard) => {
    // Use Scryfall's `/small.` thumbnail variant (~146px wide) for bandwidth.
    const thumb = item.image_uri ? item.image_uri.replace('/normal.', '/small.') : null;
    if (!thumb) return renderTextRow(item);
    return (
      <View style={s.commanderWrap}>
        {/* pointerEvents="none" wrapper keeps the image area as a bare scroll surface
            — vertical pans pass through to the FlatList responder. */}
        <View pointerEvents="none">
          <Image source={thumb} style={s.commanderThumb} contentFit="cover" cachePolicy="memory-disk" recyclingKey={thumb} />
        </View>
        {/* Label row sits beneath the image: name+mana label Pressable on the left,
            ⋮ menu button inline on the right — same plain-text treatment as the text-row
            ⋮ button so all card menus share one visual language. */}
        <View style={s.commanderLabelRow}>
          <Pressable
            onPress={() => router.push(`/card/${item.scryfall_id}`)}
            onLongPress={() => cardOptions(item)}
            delayLongPress={250}
            style={({ pressed }) => [s.commanderLabel, pressed && { opacity: 0.4 }]}
          >
            <Text style={[s.commanderName, { color: t.text }]} numberOfLines={2}>{item.name}</Text>
            <ManaCost cost={item.mana_cost} size={12} />
          </Pressable>
          <Pressable
            onPress={() => cardOptions(item)}
            hitSlop={8}
            accessibilityRole="button"
            accessibilityLabel={`Options for ${item.name}`}
            style={({ pressed }) => [s.commanderMenu, pressed && { opacity: 0.5 }]}
          >
            <Text style={[s.commanderMenuIcon, { color: t.textSecondary }]}>⋮</Text>
          </Pressable>
        </View>
      </View>
    );
  }, [router, cardOptions, t, renderTextRow]);

  // Main-deck section row — currently the standard text row. Kept as its own function so
  // future divergence (e.g., qty steppers, type icon prefixes) only touches one spot.
  const renderMainItem = useCallback((item: DeckCard) => renderTextRow(item), [renderTextRow]);

  // Sideboard section row — same shape as Main today. Separate function for the same
  // reason: lets Sideboard grow distinct affordances without disturbing Main rendering.
  const renderSideItem = useCallback((item: DeckCard) => renderTextRow(item), [renderTextRow]);

  // Considering ("maybeboard") section row — cards on the bubble that aren't in the
  // deck proper. Same row treatment as Main/Side; kept separate so the list can grow
  // its own affordances (e.g., quick "promote to main" gesture).
  const renderConsideringItem = useCallback((item: DeckCard) => renderTextRow(item), [renderTextRow]);

  // Grid tile — used in 'grid' view mode for every board (commander included). Image
  // takes the full tile width at MTG 5:7 aspect; qty badge overlays the corner when
  // the row holds 2+ copies; name truncates beneath. Tap navigates to /card/[id];
  // long-press opens the same action sheet as text rows.
  const renderGridTile = useCallback((item: DeckCard) => {
    const thumb = item.image_uri ? item.image_uri.replace('/normal.', '/small.') : null;
    return (
      <View style={{ width: tileWidth }}>
        <Pressable
          onPress={() => router.push(`/card/${item.scryfall_id}`)}
          onLongPress={() => cardOptions(item)}
          delayLongPress={250}
          style={({ pressed }) => [pressed && { opacity: 0.6 }]}
        >
          {thumb ? (
            <Image
              source={thumb}
              style={{ width: tileWidth, height: tileHeight, borderRadius: 6 }}
              contentFit="cover"
              cachePolicy="memory-disk"
              recyclingKey={thumb}
            />
          ) : (
            <View style={{ width: tileWidth, height: tileHeight, borderRadius: 6, backgroundColor: t.border }} />
          )}
        </Pressable>
        <View style={s.tileLabelRow}>
          <Text style={[s.tileName, { color: t.text }]} numberOfLines={1}>{item.name}</Text>
          <Pressable
            onPress={() => cardOptions(item)}
            hitSlop={8}
            accessibilityRole="button"
            accessibilityLabel={`Options for ${item.name}`}
            style={({ pressed }) => [s.tileMenu, pressed && { opacity: 0.5 }]}
          >
            <Text style={[s.tileMenuIcon, { color: t.textSecondary }]}>⋮</Text>
          </Pressable>
        </View>
        {/* Stepper [−][qty][+] beneath the name row — same affordance as list-view rows.
            Centered within the tile so multi-digit counts don't jitter the row. */}
        <View style={s.tileStepper}>
          <TouchableOpacity
            onPress={() => decCard(item)}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel={`Remove one ${item.name}`}
            style={[s.stepBtn, { borderColor: t.border }]}
          >
            <Text style={[s.stepIcon, { color: t.textSecondary }]}>−</Text>
          </TouchableOpacity>
          <Text style={[s.stepCount, { color: t.text }]}>{item.quantity}</Text>
          <TouchableOpacity
            onPress={() => incCard(item)}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel={`Add one ${item.name}`}
            style={[s.stepBtn, { borderColor: t.border }]}
          >
            <Text style={[s.stepIcon, { color: t.textSecondary }]}>+</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }, [router, cardOptions, incCard, decCard, tileWidth, tileHeight, t]);

  // One row of N tiles in the grid. Header rows (board / type) still render full-width
  // above each chunk so section structure is preserved.
  const renderGridRowItems = useCallback((items: DeckCard[]) => (
    <View style={[s.gridRow, { paddingHorizontal: GRID_PADDING, gap: GRID_GAP }]}>
      {items.map((item) => (
        <View key={`${item.board}:${item.scryfall_id}`}>{renderGridTile(item)}</View>
      ))}
    </View>
  ), [renderGridTile]);

  // Flatten sections into a single row stream. Each entry is either a 'header' or a 'card'
  // row — FlatList renders them as siblings, so there's NO sticky-header machinery and no
  // chance of iOS pinning a header against our wishes.
  type FlatRow =
    | { kind: 'header'; section: RowSection; key: string }
    | { kind: 'card'; item: DeckCard; key: string }
    | { kind: 'gridrow'; items: DeckCard[]; key: string };
  const flatData = useMemo<FlatRow[]>(() => {
    const out: FlatRow[] = [];
    for (const section of sections) {
      out.push({ kind: 'header', section, key: `h:${section.kind}:${section.title}` });
      // Considering rows hide when the section is collapsed; the header stays so the
      // user can re-expand it. Toggle state lives on consideringOpen.
      if (section.title === 'Considering' && !consideringOpen) continue;
      if (viewMode === 'grid') {
        // Chunk this section's cards into rows of `gridCols` tiles. Section boundaries
        // never bleed into a neighbor's chunk so each board/type still reads as a group.
        for (let i = 0; i < section.data.length; i += gridCols) {
          const items = section.data.slice(i, i + gridCols);
          out.push({ kind: 'gridrow', items, key: `g:${section.kind}:${section.title}:${i}` });
        }
      } else {
        for (const item of section.data) {
          out.push({ kind: 'card', item, key: `c:${item.board}:${item.scryfall_id}` });
        }
      }
    }
    return out;
  }, [sections, consideringOpen, viewMode, gridCols]);

  // Map each board → its header row index in flatData. Powers the speed-dial
  // nav buttons; empty entries (deck has no Considering, etc.) disable the
  // matching button instead of being hidden.
  const headerIndexByBoard = useMemo(() => {
    const m: Partial<Record<Board, number>> = {};
    flatData.forEach((row, i) => {
      if (row.kind === 'header' && row.section.kind === 'board') {
        const title = row.section.title;
        if (title === 'Commander') m.commander = i;
        else if (title === 'Main') m.main = i;
        else if (title === 'Sideboard') m.side = i;
        else if (title === 'Considering') m.considering = i;
      }
    });
    return m;
  }, [flatData]);

  // Speed-dial nav helpers. Top scrolls to the very beginning (hero band);
  // board buttons jump to that section's header row. We disable the scroll
  // animation here — FlatList's animated scroll uses a fixed platform-default
  // duration that feels sluggish for a navigation jump. Instant feels snappier
  // and matches what the user is asking the menu to do (teleport, not glide).
  // Each nav button also captures the user's current offset into savedOffset
  // so the Restore button can teleport them back where they were.
  const captureCurrentOffset = useCallback(() => {
    setSavedOffset(currentOffsetRef.current);
  }, []);
  const scrollToTop = useCallback(() => {
    captureCurrentOffset();
    listRef.current?.scrollToOffset({ offset: 0, animated: false });
    setFabOpen(false);
  }, [captureCurrentOffset]);
  const scrollToBoard = useCallback((b: Board) => {
    const idx = headerIndexByBoard[b];
    if (idx == null) return;
    captureCurrentOffset();
    listRef.current?.scrollToIndex({ index: idx, viewPosition: 0, animated: false });
    setFabOpen(false);
  }, [headerIndexByBoard, captureCurrentOffset]);
  // Restore re-uses the saved offset and clears it so the button disables
  // again until the next nav press. We don't snapshot the user's offset before
  // restoring — restoring should not itself become the "previous spot".
  const restoreOffset = useCallback(() => {
    if (savedOffset == null) return;
    listRef.current?.scrollToOffset({ offset: savedOffset, animated: false });
    setSavedOffset(null);
    setFabOpen(false);
  }, [savedOffset]);

  // FlatList row dispatcher — header rows render the section heading, card rows pick the
  // right per-board card renderer (commander tile / main row / sideboard row), gridrow
  // renders an N-tile chunk in grid view.
  const renderItem = useCallback(({ item: row }: { item: FlatRow }) => {
    if (row.kind === 'header') return renderSectionHeader(row.section);
    if (row.kind === 'gridrow') return renderGridRowItems(row.items);
    if (row.item.board === 'commander') return renderCommanderItem(row.item);
    if (row.item.board === 'side') return renderSideItem(row.item);
    if (row.item.board === 'considering') return renderConsideringItem(row.item);
    return renderMainItem(row.item);
  }, [renderSectionHeader, renderCommanderItem, renderMainItem, renderSideItem, renderConsideringItem, renderGridRowItems]);

  // NaN guard — if the route param wasn't a real number, render a tiny "not found" view.
  // Hooks above run unconditionally; queries are disabled via `enabled` so this is safe.
  if (!Number.isFinite(deckId)) {
    return (
      <View style={[s.screen, s.centered, { backgroundColor: t.bg }]}>
        <Text style={{ color: t.text, fontSize: 16, fontWeight: '600' }}>Deck not found</Text>
        <Pressable onPress={() => router.back()} style={s.backBtn}>
          <Text style={{ color: t.accent, fontSize: 14, fontWeight: '600' }}>Back</Text>
        </Pressable>
      </View>
    );
  }

  // Hard-error branch — at least one query blew up. Hide the body and offer a retry.
  if (isError) {
    return (
      <View style={[s.screen, s.centered, { backgroundColor: t.bg }]}>
        <Text style={{ color: t.text, fontSize: 16, fontWeight: '600' }}>Couldn&apos;t load deck.</Text>
        <Pressable onPress={() => { deckQ.refetch(); cardsQ.refetch(); }} style={s.backBtn}>
          <Text style={{ color: t.accent, fontSize: 14, fontWeight: '600' }}>Retry</Text>
        </Pressable>
      </View>
    );
  }

  // Deck row finished loading without error but is null/undefined — likely deleted.
  if (deckMissing) {
    return (
      <View style={[s.screen, s.centered, { backgroundColor: t.bg }]}>
        <Text style={{ color: t.text, fontSize: 16, fontWeight: '600' }}>Deck not found.</Text>
        <Pressable onPress={() => router.back()} style={s.backBtn}>
          <Text style={{ color: t.accent, fontSize: 14, fontWeight: '600' }}>Back</Text>
        </Pressable>
      </View>
    );
  }

  // Each header block in its own function — keeps the SectionList header composition
  // declarative (renderHero / renderInfoStrip / renderStatsPanel) and lets each piece
  // memoize on just its own deps.
  const renderHero = useCallback(() => (
    <DeckHero
      name={deck?.name ?? 'Deck'}
      artCropUri={deck?.art_crop_uri ?? ''}
      onBack={() => router.back()}
      onMore={more}
    />
  ), [deck?.name, deck?.art_crop_uri, router, more]);

  const renderInfoStrip = useCallback(() => (
    <DeckInfoStrip
      format={deck?.format ?? ''}
      colorIdentity={colorIdentity}
      mainCount={mainCount}
      sideCount={sideCount}
      totalPrice={totalPrice}
      statsExpanded={statsOpen}
      historyExpanded={historyOpen}
      onToggleStats={() => { setStatsOpen((v) => !v); setHistoryOpen(false); }}
      onToggleHistory={() => { setHistoryOpen((v) => !v); setStatsOpen(false); }}
    />
  ), [deck?.format, colorIdentity, mainCount, sideCount, totalPrice, statsOpen, historyOpen]);

  const renderStatsPanel = useCallback(() => (
    statsOpen ? <DeckStatsPanel main={mainCards} commander={commanderCards} side={sideCards} format={deck?.format ?? ''} /> : null
  ), [statsOpen, mainCards, commanderCards, sideCards, deck?.format]);

  const renderHistoryPanel = useCallback(() => (
    historyOpen ? <DeckHistoryPanel deckId={deckId} /> : null
  ), [historyOpen, deckId]);

  // Composed list header — hero + info strip + at most one expanded panel.
  const ListHeader = (
    <>
      {renderHero()}
      {renderInfoStrip()}
      {renderStatsPanel()}
      {renderHistoryPanel()}
    </>
  );

  // Footer slot doubles as the loading spinner / empty-state holder so it sits below
  // the last section row but above the FAB. Gated on !isLoading to avoid empty flash.
  const ListFooter = isLoading ? (
    <View style={s.loadingWrap}>
      <ActivityIndicator color={t.textSecondary} />
    </View>
  ) : sections.length === 0 ? (
    <View style={s.emptyWrap}>
      <Text style={[s.empty, { color: t.textSecondary }]}>No cards yet.</Text>
      <Text style={[s.emptySub, { color: t.textSecondary }]}>Tap + below to search Scryfall.</Text>
    </View>
  ) : null;

  return (
    // Root screen. Fills the viewport; everything inside lays out vertically.
    <View style={[s.screen, { backgroundColor: t.bg }]}>
      {/* FlatList over a flattened header+card row stream — no SectionList means no
          sticky-header behavior is possible on any platform. */}
      <FlatList
        ref={listRef}
        data={isLoading ? [] : flatData}
        keyExtractor={(row) => (row as FlatRow).key}
        renderItem={renderItem}
        ListHeaderComponent={ListHeader}
        ListFooterComponent={ListFooter}
        contentContainerStyle={s.list}
        style={{ backgroundColor: t.bg }}
        removeClippedSubviews={true}
        initialNumToRender={12}
        maxToRenderPerBatch={8}
        windowSize={7}
        // Track current offset in a ref. We don't need state — only nav-button
        // handlers read this, and they read it on press, not on each frame.
        onScroll={(e) => { currentOffsetRef.current = e.nativeEvent.contentOffset.y; }}
        scrollEventThrottle={64}
        // scrollToIndex can fire before the target row is virtualized; the
        // fallback estimates an offset from averaged item lengths so the list
        // at least lands close, after which a settle pass will land exactly.
        onScrollToIndexFailed={(info) => {
          const offset = info.averageItemLength * info.index;
          listRef.current?.scrollToOffset({ offset, animated: false });
          setTimeout(() => {
            listRef.current?.scrollToIndex({ index: info.index, viewPosition: 0, animated: false });
          }, 60);
        }}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={t.textSecondary} />}
      />

      {/* Speed-dial backdrop — only mounted while open. Tap dismisses, no
          visual treatment so the deck stays readable behind the buttons. */}
      {fabOpen && (
        <Pressable
          style={StyleSheet.absoluteFill}
          onPress={() => setFabOpen(false)}
          accessibilityRole="button"
          accessibilityLabel="Close menu"
        />
      )}

      {/* Expanded cluster: split into two anchors. Positions swapped so the
          nav column stacks directly above the FAB and the Add-card pill sits
          to the LEFT of the FAB at FAB level. */}
      {fabOpen && (
        <>
          <View
            style={[s.dialAddCard, { bottom: 16 + insets.bottom }]}
            pointerEvents="box-none"
          >
            <DialButton t={t} label="Add card" onPress={openAdd} primary />
          </View>
          <View
            style={[s.dialNavCol, { bottom: 16 + insets.bottom + 56 + 8 }]}
            pointerEvents="box-none"
          >
            {/* Restore sits at the top of the column so it's visually distinct
                from the "jump-to" entries below it. Disabled until a nav button
                has captured a previous offset. */}
            <DialButton t={t} label="Prev" onPress={restoreOffset} disabled={savedOffset == null} />
            <DialButton t={t} label="Top" onPress={scrollToTop} />
            <DialButton t={t} label="Main" onPress={() => scrollToBoard('main')} disabled={headerIndexByBoard.main == null} />
            <DialButton t={t} label="Sideboard" onPress={() => scrollToBoard('side')} disabled={headerIndexByBoard.side == null} />
            <DialButton t={t} label="Considering" onPress={() => scrollToBoard('considering')} disabled={headerIndexByBoard.considering == null} />
          </View>
        </>
      )}

      {/* Floating Action Button — toggles the speed-dial. Glyph flips to ×
          while open so the close affordance is obvious. */}
      <Pressable
        onPress={() => setFabOpen((v) => !v)}
        style={[s.fab, { backgroundColor: t.accent, bottom: 16 + insets.bottom }]}
        accessibilityRole="button"
        accessibilityLabel={fabOpen ? 'Close menu' : 'Open menu'}
        accessibilityState={{ expanded: fabOpen }}
      >
        <Text style={s.fabLabel}>{fabOpen ? '×' : '+'}</Text>
      </Pressable>

      {/* In-page card search + add (no nav to /search). Sheet stays open across adds. */}
      <AddCardsSheet
        visible={addOpen}
        deckId={deckId}
        format={deck?.format ?? ''}
        commanderColorIdentity={commanderColorIdentity}
        onClose={() => setAddOpen(false)}
        onInspect={() => { resumeAddOnFocus.current = true; }}
      />
      {/* Printing-picker sheet — opened from the per-card action sheet's
          "Change printing" entry. Mounted unconditionally so the slide
          animation runs on first show. */}
      <ChangePrintingSheet
        visible={printingTarget !== null}
        card={printingTarget}
        deckId={deckId}
        onClose={() => setPrintingTarget(null)}
      />
      {/* Generic action sheet portal — replaces native Alert across the app. */}
      {sheet.node}
    </View>
  );
}

// Single pill button used inside the speed-dial. `primary` paints with the
// accent (used for "Add card" so it stands out from the nav column). Disabled
// state dims the pill — used when a board has no header in the deck.
function DialButton({
  t, label, onPress, primary, disabled,
}: {
  t: ReturnType<typeof useTheme>;
  label: string;
  onPress: () => void;
  primary?: boolean;
  disabled?: boolean;
}) {
  const bg = primary ? t.accent : t.surface;
  const fg = primary ? '#fff' : t.text;
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled: !!disabled }}
      style={({ pressed }) => [
        s.dialBtn,
        { backgroundColor: bg, borderColor: t.border },
        disabled && { opacity: 0.4 },
        pressed && !disabled && { opacity: 0.7 },
      ]}
    >
      <Text style={[s.dialBtnText, { color: fg }]}>{label}</Text>
    </Pressable>
  );
}

const s = StyleSheet.create({
  // Root container — fills the viewport.
  screen: { flex: 1 },
  // ScrollView content — 100px bottom pad so the FAB never overlaps the last row.
  list: { paddingBottom: 100 },
  // Generic section header (used for both board headers and type sub-section headers).
  section: { paddingHorizontal: 14, paddingTop: 8, paddingBottom: 4, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  // Board header overlay: extra top spacing + a hairline divider above to separate boards.
  boardSection: { paddingTop: 16, marginTop: 4, borderTopWidth: 1 },
  // Bold uppercase title for boards (Commander / Main / Sideboard).
  boardTitle: { fontSize: 13, fontWeight: '800', textTransform: 'uppercase', letterSpacing: 0.8 },
  // Smaller uppercase title for type sub-sections inside Main (Creatures / Lands / etc.).
  typeTitle: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.6 },
  // Right-aligned section count (and optional · $price for boards).
  sectionCount: { fontSize: 12, fontWeight: '600' },
  // Standard text row container: holds the main pressable area + trailing ⋮ menu button.
  // Hairline separator between rows lives on the outer wrap so the menu button shares it.
  row: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 14, borderBottomWidth: StyleSheet.hairlineWidth },
  // Inner main-row Pressable — the qty/mana/name/price content. Sized as flex:1 so the
  // ⋮ button sits flush right.
  rowMain: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 12 },
  // ⋮ menu tap target — sits at the LEFT of the row now. Generous left padding
  // keeps the icon clear of the phone's curved bevel; right padding gives breathing
  // room before the row's main pressable begins.
  rowMenu: { paddingVertical: 12, paddingLeft: 4, paddingRight: 10 },
  rowMenuIcon: { fontSize: 18, fontWeight: '700' },
  // Inline qty stepper — [−][N][+]. Sits at the trailing edge of the row now.
  // marginLeft pulls it away from the row's price/name content; the row container's
  // own paddingHorizontal keeps the rightmost button off the bevel.
  stepper: { flexDirection: 'row', alignItems: 'center', gap: 4, marginLeft: 6 },
  stepBtn: {
    width: 24, height: 24, borderRadius: 6, borderWidth: StyleSheet.hairlineWidth,
    alignItems: 'center', justifyContent: 'center',
  },
  stepIcon: { fontSize: 16, fontWeight: '700', lineHeight: 18, includeFontPadding: false } as const,
  stepCount: { minWidth: 18, textAlign: 'center', fontSize: 13, fontWeight: '700', fontVariant: ['tabular-nums'] },
  // Commander row outer wrap — full width, centers its children. NO touch handlers
  // anywhere on the image so the entire commander tile area is a bare scroll surface.
  commanderWrap: { alignItems: 'center', paddingVertical: 12, paddingHorizontal: 14, gap: 8 },
  // Commander label Pressable — sits inside the label row beside the ⋮ button. flex:1
  // so it fills the space left of the menu; centered text/mana stack inside.
  commanderLabel: { flex: 1, alignItems: 'center', gap: 4, paddingVertical: 8, paddingHorizontal: 12 },
  // Commander name — explicit max width, NO flex:1. The shared `s.name` is for horizontal
  // rows where flex:1 fills remaining space; reusing it here in a column container makes
  // yoga grow the Text unboundedly and breaks FlatList row measurement.
  commanderName: { maxWidth: 220, fontSize: 14, fontWeight: '600', textAlign: 'center' },
  // Commander art thumbnail — MTG card aspect (5:7), 220×308px. Fixed height (not
  // aspectRatio) so FlatList can measure the row up-front; aspectRatio lets yoga delay
  // height resolution which can stall VirtualizedList scroll past this row.
  commanderThumb: { width: 220, height: 308, borderRadius: 8 },
  // Horizontal label row beneath the image — holds the name+mana label and the inline
  // ⋮ menu button. Capped at the image width so the button sits at the tile's edge.
  commanderLabelRow: { flexDirection: 'row', alignItems: 'center', maxWidth: 220, width: 220 },
  // Inline ⋮ button — plain text, no chip/circle, mirrors text-row treatment for
  // consistency. Slightly larger glyph (22) than rowMenuIcon (18) to match the
  // commander tile's bigger visual weight. Right pad sized to mirror rowMenu so both
  // menus sit the same distance from the screen edge.
  commanderMenu: { paddingVertical: 8, paddingLeft: 8, paddingRight: 12 },
  commanderMenuIcon: { fontSize: 22, fontWeight: '700' },
  // Card name — fills remaining row width with single-line truncation.
  name: { flex: 1, fontSize: 13, fontWeight: '600' },
  // Trailing per-card USD price — right-aligned, dim, fixed width so the column
  // stays vertically aligned across rows even when names are short or wrap.
  price: { minWidth: 52, textAlign: 'right', fontSize: 11, fontVariant: ['tabular-nums'], fontWeight: '600' },
  // Empty-state wrapper centers two stacked text lines with a top offset.
  emptyWrap: { alignItems: 'center', marginTop: 60 },
  // Empty-state primary line.
  empty: { textAlign: 'center' },
  // Empty-state sub-line — quieter call-to-action under the primary line.
  emptySub: { textAlign: 'center', fontSize: 12, marginTop: 4, opacity: 0.85 },
  // Centered fallback view (NaN guard, error, deck-missing) — fills the screen.
  centered: { alignItems: 'center', justifyContent: 'center' },
  // Back/Retry button below the fallback message.
  backBtn: { marginTop: 12, paddingHorizontal: 16, paddingVertical: 8 },
  // Loading spinner well — slim vertical band where sections would have rendered.
  loadingWrap: { paddingVertical: 32, alignItems: 'center' },
  // Floating Action Button — bottom-right corner, accent fill, soft shadow.
  fab: {
    position: 'absolute', right: 24, width: 56, height: 56, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center', elevation: 6,
    shadowColor: '#000', shadowOpacity: 0.20, shadowRadius: 6, shadowOffset: { width: 0, height: 4 },
  },
  // FAB plus icon.
  fabLabel: { color: 'white', fontSize: 28, fontWeight: '300', marginTop: -2 },
  // Add-card pill: bottom-aligned with the FAB, sitting to the LEFT of it.
  // Right offset clears the FAB's 56pt width plus a small gap.
  dialAddCard: { position: 'absolute', right: 24 + 56 + 10, alignItems: 'flex-end' },
  // Nav column: stacked directly above the FAB, right-aligned with it.
  dialNavCol: { position: 'absolute', right: 24, alignItems: 'flex-end', gap: 8 },
  // Pill button shared by every entry. Width is intrinsic to the label so the
  // column auto-sizes to its widest member.
  dialBtn: {
    paddingHorizontal: 14, paddingVertical: 10, borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
    elevation: 4, shadowColor: '#000', shadowOpacity: 0.18, shadowRadius: 4, shadowOffset: { width: 0, height: 2 },
  },
  dialBtnText: { fontSize: 13, fontWeight: '700' },
  // Grid view: row of N tiles. Outer paddingHorizontal mirrors text-row inset so grid
  // and list views align horizontally; gap supplied inline (depends on screen width).
  gridRow: { flexDirection: 'row', paddingVertical: 6 },
  // Tile label row — name (flex:1, truncates) + inline ⋮ menu button. Sits beneath
  // the image; same plain-text ⋮ treatment as the text-row and commander menus.
  tileLabelRow: { flexDirection: 'row', alignItems: 'center', marginTop: 4, gap: 4 },
  // Per-tile name — single line, dim. flex:1 so the ⋮ button sits flush to the
  // tile's right edge while the name truncates within the remaining width.
  tileName: { flex: 1, fontSize: 11, fontWeight: '600', textAlign: 'center' },
  tileMenu: { paddingHorizontal: 4, paddingVertical: 2 },
  tileMenuIcon: { fontSize: 14, fontWeight: '700', lineHeight: 16, includeFontPadding: false } as const,
  // Stepper row beneath the name — centered [−][qty][+] flush within tile width.
  tileStepper: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, marginTop: 4 },
});
