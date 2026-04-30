import { useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList, KeyboardAvoidingView, Modal, Platform,
  Pressable, StyleSheet, Text, TextInput, TouchableOpacity, View,
} from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { addCardToDeck, decrementCardInDeck, ensureDeckArt, getDeckCards, type Board } from '../db/decks';
import { upsertCard, type CachedCard } from '../db/cards';
import { searchScryfall } from '../api/scryfall';
import { useKeyboardAppearance, useTheme } from '../theme/useTheme';
import { useStore, type AddSheetCardSize } from '../store/useStore';
import { Skeleton } from './Skeleton';
import { ManaCost } from './ManaCost';

const BOARDS: { key: Board; label: string }[] = [
  { key: 'main', label: 'Main' },
  { key: 'side', label: 'Side' },
  { key: 'commander', label: 'Cmdr' },
  { key: 'considering', label: 'Maybe' },
];

type CardFace = { name?: string; oracle_text?: string };

// MTG card aspect is 5:7. small/medium stay inside Scryfall's /small. source
// (146×204). large is "full card" mode — image fills the row and no metadata
// renders, so the printed card text is the read surface. We use the /normal.
// variant at large so the bigger render stays sharp.
const CARD_SIZE: Record<AddSheetCardSize, { w: number; h: number }> = {
  small: { w: 40, h: 56 },
  medium: { w: 64, h: 90 },
  large: { w: 220, h: 308 },
};
const CARD_SIZE_CYCLE: AddSheetCardSize[] = ['small', 'medium', 'large'];
// Pill shows the single-letter abbreviation; full word is the source-of-truth
// internal name.
const CARD_SIZE_LABEL: Record<AddSheetCardSize, string> = {
  small: 'S', medium: 'M', large: 'L',
};
// Coerce any persisted/legacy value to a known key so old store snapshots
// (sm/md/lg/xl, compact/normal/full) don't crash CARD_SIZE lookups.
function normalizeSize(v: unknown): AddSheetCardSize {
  if (v === 'small' || v === 'medium' || v === 'large') return v;
  if (v === 'compact' || v === 'sm') return 'small';
  if (v === 'full' || v === 'xl') return 'large';
  return 'medium';
}

/**
 * Oracle text for the result row. For multi-face cards (DFC / transform /
 * modal_dfc / split / flip / adventure / etc.) we walk `card_faces[]` and
 * stitch each face's name + oracle_text together so the user sees both halves
 * on expand. Meld is excluded — its top-level `oracle_text` is already the
 * canonical reading and the `all_parts` graph isn't shaped for inline display.
 */
function oracleTextFor(card: CachedCard): string {
  if (card.layout !== 'meld') {
    try {
      const faces = JSON.parse(card.card_faces ?? '[]') as CardFace[];
      if (Array.isArray(faces) && faces.length > 1) {
        const parts = faces
          .map((f) => {
            const name = (f.name ?? '').trim();
            const text = (f.oracle_text ?? '').trim();
            if (!text) return '';
            return name ? `${name}\n${text}` : text;
          })
          .filter(Boolean);
        if (parts.length) return parts.join('\n\n—\n\n');
      }
    } catch { /* malformed JSON → fall through to top-level text */ }
  }
  return (card.oracle_text ?? '').trim();
}

type Props = {
  visible: boolean;
  deckId: number;
  /** Deck's format (e.g. 'commander', 'modern'). When set, search queries are
      prefixed with `legal:<format>` so results are filtered to legal printings.
      Empty string disables the filter. */
  format?: string;
  /** Commander deck only: lowercased WUBRG (or 'c' for colorless) string of
      the deck's commander color identity. When provided + format=commander
      we use Scryfall's `commander:<ci>` shorthand (subset-of color identity
      AND legal:commander) instead of plain `legal:commander`. */
  commanderColorIdentity?: string;
  onClose: () => void;
  /** Called right before the sheet closes to navigate to a card detail. The parent
      uses this to set a "resume on focus" flag so the sheet auto-reopens with the
      same search when the user returns from the detail screen. */
  onInspect?: () => void;
};

export function AddCardsSheet({ visible, deckId, format, commanderColorIdentity, onClose, onInspect }: Props) {
  const t = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const qc = useQueryClient();
  const cardSize = useStore((s) => normalizeSize(s.addSheetCardSize));
  const setCardSize = useStore((s) => s.setAddSheetCardSize);
  const cycleCardSize = () => {
    const i = CARD_SIZE_CYCLE.indexOf(cardSize);
    setCardSize(CARD_SIZE_CYCLE[(i + 1) % CARD_SIZE_CYCLE.length]);
  };
  const thumbDim = CARD_SIZE[cardSize];
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const inputRef = useRef<TextInput>(null);

  const [board, setBoard] = useState<Board>('main');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<CachedCard[]>([]);
  const [loading, setLoading] = useState(false);
  // recent stores the names+ids of cards just added so the user can re-add a 2nd/3rd
  // copy via a single tap on the chip strip without re-searching.
  const [recent, setRecent] = useState<CachedCard[]>([]);
  // Per-row expand toggles. Tapping a row toggles its scryfall_id in this set;
  // expanded rows show the full oracle_text instead of a 2-line truncation.
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());
  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  // Normalize the deck's format into a Scryfall query prefix. For commander
  // decks with a known color identity we combine the explicit format-legality
  // check with the color-identity subset filter — `(legal:commander and
  // commander:<ci>)`. The `commander:` shorthand already implies
  // `legal:commander`, but spelling both out makes the constructed query
  // self-documenting in the live preview. For everything else we use plain
  // `legal:<format>`. Empty/blank format → no prefix.
  const legalPrefix = useMemo(() => {
    const f = (format ?? '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    if (!f) return '';
    if (f === 'commander' && commanderColorIdentity) {
      return `(legal:commander and commander:${commanderColorIdentity}) `;
    }
    return `legal:${f} `;
  }, [format, commanderColorIdentity]);

  // Search-as-you-type — 150ms debounce instead of 300ms. Feels closer to instant
  // without stampeding Scryfall. Aborts in-flight on every keystroke.
  useEffect(() => {
    if (!visible) return;
    const q = query.trim();
    if (q.length < 2) { setResults([]); setLoading(false); return; }
    const ctrl = new AbortController();
    setLoading(true);
    const timer = setTimeout(() => {
      searchScryfall(`${legalPrefix}${q}`, 1, ctrl.signal)
        .then((r) => { setResults(r); setLoading(false); })
        .catch(() => { if (!ctrl.signal.aborted) setLoading(false); });
    }, 150);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [query, visible, legalPrefix]);

  // New search → collapse any previously expanded rows; their ids are no
  // longer in `results` but the set would otherwise persist across queries.
  useEffect(() => { setExpandedIds(new Set()); }, [results]);

  // Refocus the input every time the sheet opens so the keyboard is up immediately.
  // Intentionally do NOT clear `query`/`results` on close — when the user taps a card
  // to inspect (which closes the sheet to reveal the detail screen), navigating back
  // should restore the same search they were working in.
  useEffect(() => {
    if (visible) {
      const id = setTimeout(() => inputRef.current?.focus(), 30);
      return () => clearTimeout(id);
    }
  }, [visible]);

  // Live deck contents — used to decide whether the "−" button should disable
  // (no copies on the active board) and to render the per-row qty hint. Reuses
  // the same query key the deck screen uses, so this view shares its cache.
  const deckCardsQ = useQuery({
    queryKey: ['deck-cards', deckId],
    queryFn: () => getDeckCards(deckId),
    enabled: visible && Number.isFinite(deckId),
  });
  // Map "name|board" → { qty, ids }. Keyed on lowercased card name (not
  // scryfall_id) so alternate printings count as the same card — pressing −
  // on a Sol Ring from set X removes a copy of any Sol Ring already in the
  // deck on that board. `ids` collects every scryfall_id contributing to the
  // count so `remove()` can pick a real row to decrement.
  const qtyByName = useMemo(() => {
    const m = new Map<string, { qty: number; ids: string[] }>();
    for (const c of deckCardsQ.data ?? []) {
      const key = `${c.name.toLowerCase()}|${c.board}`;
      const cur = m.get(key);
      if (cur) { cur.qty += c.quantity; cur.ids.push(c.scryfall_id); }
      else m.set(key, { qty: c.quantity, ids: [c.scryfall_id] });
    }
    return m;
  }, [deckCardsQ.data]);
  const qtyOnBoard = (name: string) => qtyByName.get(`${name.toLowerCase()}|${board}`)?.qty ?? 0;

  const invalidateAfterMutation = () => {
    qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
    qc.invalidateQueries({ queryKey: ['deck', deckId] });
    qc.invalidateQueries({ queryKey: ['decks'] });
    qc.invalidateQueries({ queryKey: ['deck-history', deckId] });
  };

  const add = (card: CachedCard) => {
    upsertCard(card);
    addCardToDeck({ deck_id: deckId, scryfall_id: card.scryfall_id, quantity: 1, board });
    invalidateAfterMutation();
    void ensureDeckArt(deckId, card.scryfall_id).then(() => {
      qc.invalidateQueries({ queryKey: ['deck', deckId] });
      qc.invalidateQueries({ queryKey: ['decks'] });
    });
    setRecent((r) => {
      const next = [card, ...r.filter((c) => c.scryfall_id !== card.scryfall_id)];
      return next.slice(0, 6);
    });
  };

  // Remove one copy from the active board. Alt-printings of the same card
  // count as one entity (qtyByName is keyed on name), so we look up which
  // scryfall_id is actually in the deck on this board and decrement that.
  // No-op when no copies of any printing exist there.
  const remove = (card: CachedCard) => {
    const entry = qtyByName.get(`${card.name.toLowerCase()}|${board}`);
    if (!entry || entry.qty <= 0) return;
    // Pick the first id present on this board — any printing works for a
    // decrement, and `decrementCardInDeck` deletes the row at qty 0.
    decrementCardInDeck(deckId, entry.ids[0], board);
    invalidateAfterMutation();
  };

  // Tapping a result's main row navigates to the card detail page. We must
  // dismiss this RN Modal before pushing the navigator's modal route — RN's
  // <Modal> is presented at window level on iOS, so leaving it mounted hides
  // the destination screen behind it. Closing and pushing in the same tick
  // racing on iOS leaves a stuck touch interceptor on the deck screen after
  // the user returns (taps land on nothing, gestures still work). Deferring
  // the push by a frame lets the Modal commit its dismissal first.
  const inspect = (card: CachedCard) => {
    upsertCard(card);
    onInspect?.();
    onClose();
    requestAnimationFrame(() => {
      router.push(`/card/${card.scryfall_id}`);
    });
  };

  // Pre-build per-result thumbnail URI. At xl we keep the /normal. variant
  // since the rendered card is large enough to need the higher-res source.
  const thumbFor = (c: CachedCard): string | null => {
    if (!c.image_uri) return null;
    return cardSize === 'large' ? c.image_uri : c.image_uri.replace('/normal.', '/small.');
  };

  // Sized so the card sits in the upper third — paddingTop pushes it well clear of
  // the keyboard regardless of keyboard height. Bottom max-height lets it grow up to
  // (but not past) the keyboard's top edge.
  const containerPadTop = useMemo(() => insets.top + 60, [insets.top]);

  return (
    <Modal visible={visible} transparent animationType="none" hardwareAccelerated onRequestClose={onClose} statusBarTranslucent>
      {/* Backdrop tap dismisses. */}
      <Pressable style={s.backdrop} onPress={onClose}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={[s.anchor, { paddingTop: containerPadTop }]}
        >
          {/* Card stops bubble propagation so taps inside don't dismiss.
              Background alpha (~92%) keeps the deck faintly visible behind so
              the search overlay doesn't feel like a fully separate screen. */}
          <Pressable
            onPress={(e) => e.stopPropagation()}
            style={[s.card, { backgroundColor: t.surface + 'EB', borderColor: t.border }]}
          >
            {/* Header: Add to [chips] · ✕ */}
            <View style={s.headerRow}>
              <Text style={[s.headerLabel, { color: t.textSecondary }]}>Add to</Text>
              <View style={s.boardChips}>
                {BOARDS.map((b) => (
                  <Pressable key={b.key} onPress={() => setBoard(b.key)}
                    style={[s.boardChip, { backgroundColor: b.key === board ? t.accent : t.surfaceAlt }]}>
                    <Text style={[s.boardChipText, { color: b.key === board ? '#fff' : t.text }]}>{b.label}</Text>
                  </Pressable>
                ))}
              </View>
              {/* Size toggle — single tap cycles S → M → L → S. Button face
                  shows the active letter so the current size is always
                  visible without tapping to discover. */}
              <Pressable
                onPress={cycleCardSize}
                hitSlop={8}
                accessibilityRole="button"
                accessibilityLabel={`Card size: ${cardSize}. Tap to cycle.`}
                style={[s.sizeBtn, { backgroundColor: t.surfaceAlt, borderColor: t.border }]}
              >
                <Text style={[s.sizeBtnText, { color: t.text }]}>{CARD_SIZE_LABEL[cardSize]}</Text>
              </Pressable>
              <Pressable onPress={onClose} hitSlop={10} style={s.close}>
                <Text style={[s.closeIcon, { color: t.textSecondary }]}>✕</Text>
              </Pressable>
            </View>

            {/* Big search input. Submit (return) adds the top result. */}
            <View style={[s.inputWrap, { backgroundColor: t.bg, borderColor: t.border }]}>
              <Text style={[s.inputIcon, { color: t.textSecondary }]}>⌕</Text>
              <TextInput
                ref={inputRef}
                value={query}
                onChangeText={setQuery}
                placeholder="Search Scryfall…"
                placeholderTextColor={t.textSecondary}
                returnKeyType="done"
                autoCorrect={false}
                autoCapitalize="none"
                keyboardAppearance={keyboardAppearance}
                style={[s.input, { color: t.text }]}
              />
              {query.length > 0 ? (
                <Pressable onPress={() => setQuery('')} hitSlop={8} style={s.clear}>
                  <Text style={[s.clearIcon, { color: t.textSecondary }]}>×</Text>
                </Pressable>
              ) : null}
            </View>

            {/* Live preview of the actual Scryfall query string. Lets the
                user see the format/commander filters their typed text is
                being prepended with. Hidden when there's nothing to show. */}
            {(legalPrefix || query.trim()) ? (
              <Text style={[s.queryPreview, { color: t.textSecondary }]} numberOfLines={1}>
                <Text style={{ color: t.accent }}>{legalPrefix}</Text>{query.trim()}
              </Text>
            ) : null}

            {/* Quick-add strip — chips of cards added this session, tap to add another copy. */}
            {recent.length > 0 ? (
              <View style={s.recentWrap}>
                <Text style={[s.recentLabel, { color: t.textSecondary }]}>Just added</Text>
                <FlatList
                  data={recent}
                  keyExtractor={(c) => c.scryfall_id}
                  horizontal
                  keyboardShouldPersistTaps="handled"
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={{ gap: 6 }}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      onPress={() => add(item)}
                      style={[s.recentChip, { backgroundColor: t.surfaceAlt, borderColor: t.border }]}
                    >
                      <Text style={[s.recentChipText, { color: t.text }]} numberOfLines={1}>
                        +1 {item.name}
                      </Text>
                    </TouchableOpacity>
                  )}
                />
              </View>
            ) : null}

            {/* Result body */}
            <View style={s.resultsBody}>
              {loading ? (
                // Skeleton rows mirror the final result-row shape (thumb,
                // name, mana/type, oracle snippet) so the list doesn't jump
                // when results land. Count is intentionally below the screen
                // so it never crowds; result list will scroll as needed.
                <View style={{ paddingTop: 4 }}>
                  {Array.from({ length: 6 }).map((_, i) => (
                    <View key={i} style={s.resultRow}>
                      <View style={s.resultMain}>
                        <Skeleton width={thumbDim.w} height={thumbDim.h} radius={4} />
                        <View style={[s.resultMeta, { gap: 6 }]}>
                          <Skeleton height={14} width="70%" />
                          <Skeleton height={11} width="50%" />
                          <Skeleton height={11} width="90%" />
                        </View>
                      </View>
                    </View>
                  ))}
                </View>
              ) : query.trim().length < 2 ? (
                <Text style={[s.hint, { color: t.textSecondary }]}>Type 2+ characters to search.</Text>
              ) : results.length === 0 ? (
                <Text style={[s.hint, { color: t.textSecondary }]}>No matches.</Text>
              ) : (
                <FlatList
                  data={results}
                  keyExtractor={(c) => c.scryfall_id}
                  keyboardShouldPersistTaps="handled"
                  removeClippedSubviews
                  initialNumToRender={16}
                  maxToRenderPerBatch={10}
                  windowSize={9}
                  ItemSeparatorComponent={() => <View style={[s.sep, { backgroundColor: t.border }]} />}
                  renderItem={({ item }) => {
                    const thumb = thumbFor(item);
                    const expanded = expandedIds.has(item.scryfall_id);
                    const oracle = oracleTextFor(item);
                    // large mode: full card image stands alone with stepper
                    // buttons beside it. Card text is read off the printed
                    // image, so we hide name/mana/type/oracle entirely.
                    if (cardSize === 'large') {
                      const qty = qtyOnBoard(item.scryfall_id);
                      const canRemove = qty > 0;
                      return (
                        <View style={s.resultRowXl}>
                          <Pressable
                            onPress={() => inspect(item)}
                            onLongPress={() => inspect(item)}
                            style={({ pressed }) => [pressed && { opacity: 0.7 }]}
                            accessibilityLabel={`Inspect ${item.name}`}
                          >
                            {thumb ? (
                              <Image
                                source={thumb}
                                style={[s.resultThumb, { width: thumbDim.w, height: thumbDim.h }]}
                                contentFit="cover"
                                priority="low"
                                decodeFormat="rgb"
                                recyclingKey={thumb}
                              />
                            ) : (
                              <View style={[s.resultThumb, { width: thumbDim.w, height: thumbDim.h, backgroundColor: t.surfaceAlt }]} />
                            )}
                          </Pressable>
                          {/* Stepper floats to the right edge so the card
                              itself stays centered in the row. */}
                          <View style={s.xlStepper}>
                            <TouchableOpacity
                              onPress={() => remove(item)}
                              disabled={!canRemove}
                              hitSlop={6}
                              style={[
                                s.qtyBtn,
                                { borderColor: t.border },
                                !canRemove && { opacity: 0.35 },
                              ]}
                              accessibilityLabel={`Remove one ${item.name}`}
                            >
                              <Text style={[s.qtyBtnText, { color: t.text }]}>−</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                              onPress={() => add(item)}
                              hitSlop={6}
                              style={[s.qtyBtn, s.qtyBtnPrimary, { backgroundColor: t.accent }]}
                              accessibilityLabel={`Add one ${item.name}`}
                            >
                              <Text style={s.qtyBtnTextPrimary}>+</Text>
                            </TouchableOpacity>
                          </View>
                        </View>
                      );
                    }
                    return (
                      <View style={s.resultRow}>
                        {/* Tap row → toggle inline oracle expansion. Long-press →
                            inspect (full card detail screen). 350ms delay leaves
                            the tap snappy while still distinguishing intent. */}
                        <Pressable
                          onPress={() => toggleExpand(item.scryfall_id)}
                          onLongPress={() => inspect(item)}
                          delayLongPress={350}
                          style={({ pressed }) => [s.resultMain, pressed && { opacity: 0.6 }]}
                          accessibilityLabel={`${item.name}. Tap to expand, long press to inspect.`}
                        >
                          {thumb ? (
                            <Image
                              source={thumb}
                              style={[s.resultThumb, { width: thumbDim.w, height: thumbDim.h }]}
                              contentFit="cover"
                              priority="low"
                              decodeFormat="rgb"
                              recyclingKey={thumb}
                            />
                          ) : (
                            <View style={[s.resultThumb, { width: thumbDim.w, height: thumbDim.h, backgroundColor: t.surfaceAlt }]} />
                          )}
                          <View style={s.resultMeta}>
                            <View style={s.resultTopRow}>
                              <Text style={[s.resultName, { color: t.text }]} numberOfLines={1}>{item.name}</Text>
                            </View>
                            <View style={s.resultSubRow}>
                              <ManaCost cost={item.mana_cost} size={11} />
                              <Text style={[s.resultType, { color: t.textSecondary }]} numberOfLines={1}>
                                {item.type_line ?? ''}
                              </Text>
                              {/* Chevron cues that the row toggles. ▾ collapsed
                                  → more text available; ▴ expanded → tap to
                                  collapse. Hidden when there's no oracle text
                                  to expand at all. */}
                              {oracle ? (
                                <Text style={[s.expandIcon, { color: t.textSecondary }]}>
                                  {expanded ? '▴' : '▾'}
                                </Text>
                              ) : null}
                            </View>
                            {/* Oracle snippet — 2 lines collapsed, full text when
                                expanded. Hidden entirely if the card has none. */}
                            {oracle ? (
                              <Text
                                style={[s.resultOracle, { color: t.textSecondary }]}
                                numberOfLines={expanded ? undefined : 2}
                              >
                                {oracle}
                              </Text>
                            ) : null}
                          </View>
                        </Pressable>
                        {/* − removes one copy from the active board (disabled when none).
                            + adds one copy. Stepper-style controls so power users build
                            decks without leaving the row. */}
                        {(() => {
                          const qty = qtyOnBoard(item.name);
                          const canRemove = qty > 0;
                          return (
                            <>
                              <TouchableOpacity
                                onPress={() => remove(item)}
                                disabled={!canRemove}
                                hitSlop={6}
                                style={[
                                  s.qtyBtn,
                                  { borderColor: t.border },
                                  !canRemove && { opacity: 0.35 },
                                ]}
                                accessibilityLabel={`Remove one ${item.name}`}
                                accessibilityState={{ disabled: !canRemove }}
                              >
                                <Text style={[s.qtyBtnText, { color: t.text }]}>−</Text>
                              </TouchableOpacity>
                              <TouchableOpacity
                                onPress={() => add(item)}
                                hitSlop={6}
                                style={[s.qtyBtn, s.qtyBtnPrimary, { backgroundColor: t.accent }]}
                                accessibilityLabel={`Add one ${item.name}`}
                              >
                                <Text style={s.qtyBtnTextPrimary}>+</Text>
                              </TouchableOpacity>
                            </>
                          );
                        })()}
                      </View>
                    );
                  }}
                />
              )}
            </View>
          </Pressable>
        </KeyboardAvoidingView>
      </Pressable>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)' },
  // Anchor pulls the card to the TOP of the visible area so it sits well clear of
  // the keyboard. KeyboardAvoidingView pads from the bottom on iOS as the keyboard rises.
  anchor: { flex: 1, alignItems: 'center', justifyContent: 'flex-start' },
  // Card body — rounded all sides, fixed-ish max height that accommodates results
  // without crowding the keyboard. Width capped for tablet but full-width-ish on phones.
  card: {
    // flex:1 so the card claims all space the KeyboardAvoidingView gives it (down to
    // wherever the keyboard ends). Without flex:1 the card sizes to its content, which
    // collapses `resultsBody flex:1` to its minHeight (80pt = ~1 row).
    // marginBottom keeps a small gap above the keyboard's top edge.
    flex: 1, width: '94%', maxWidth: 520, marginBottom: 12,
    borderRadius: 16, borderWidth: 1, padding: 14, gap: 10,
    shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 12, shadowOffset: { width: 0, height: 6 },
    elevation: 8,
  },

  // ---- Header ----
  headerRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headerLabel: { fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6 },
  boardChips: { flex: 1, flexDirection: 'row', gap: 6 },
  boardChip: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999 },
  boardChipText: { fontSize: 12, fontWeight: '600' },
  close: { width: 32, height: 32, alignItems: 'center', justifyContent: 'center' },
  closeIcon: { fontSize: 18, fontWeight: '600' },
  // Size toggle pill — sits between the board chips and the close button.
  // Compact pill with a single-letter face so the header doesn't grow.
  sizeBtn: {
    minWidth: 28, height: 28, paddingHorizontal: 8, borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth, alignItems: 'center', justifyContent: 'center',
  },
  sizeBtnText: { fontSize: 12, fontWeight: '700' },

  // ---- Search input ----
  inputWrap: {
    flexDirection: 'row', alignItems: 'center',
    height: 48, borderRadius: 12, borderWidth: 1, paddingHorizontal: 12, gap: 8,
  },
  inputIcon: { fontSize: 18 },
  input: { flex: 1, fontSize: 16, paddingVertical: 0 },
  clear: { width: 24, height: 24, alignItems: 'center', justifyContent: 'center' },
  clearIcon: { fontSize: 20, fontWeight: '700' },
  // Live Scryfall query preview — monospaced so the prefix/query syntax reads
  // like a search string. Truncates to one line so a long query doesn't shove
  // the results body down.
  queryPreview: { fontSize: 11, fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) },

  // ---- Recent strip ----
  recentWrap: { gap: 4 },
  recentLabel: { fontSize: 10, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6 },
  recentChip: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, borderWidth: 1, maxWidth: 200 },
  recentChipText: { fontSize: 12, fontWeight: '600' },

  // ---- Results list ----
  // flex:1 to fill all space below header/input/recent. Removed minHeight cap so it
  // can't accidentally pin to a tiny height — needs the card itself to be flex:1.
  resultsBody: { flex: 1 },
  hint: { fontSize: 13, textAlign: 'center', paddingVertical: 24 },
  sep: { height: StyleSheet.hairlineWidth },
  // Tighter row padding (was 8) packs ~30% more rows in the same vertical space.
  resultRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 5 },
  // full-size row — card image is centered in the row, stepper floats
  // absolutely on the right edge so it doesn't push the card off-center.
  resultRowXl: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 10 },
  xlStepper: { position: 'absolute', right: 8, top: 0, bottom: 0, flexDirection: 'column', justifyContent: 'center', gap: 10 },
  resultMain: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 10 },
  // Thumbnail container — actual width/height applied per-render from the
  // user's selected size in the store. /small. source is 146×204 so even the
  // largest preset stays inside source dims (no upscaling artifacts).
  resultThumb: { borderRadius: 4 },
  resultMeta: { flex: 1, gap: 2 },
  resultTopRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  resultName: { flex: 1, fontSize: 14, fontWeight: '600' },
  resultSubRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  resultType: { flex: 1, fontSize: 11 },
  // Oracle snippet — italics + slightly smaller than the type line so it reads
  // as supplementary text. Line-height bumped so multi-line wraps are legible.
  resultOracle: { fontSize: 11, lineHeight: 15, marginTop: 2 },
  // Expand/collapse chevron — sits on the right edge of the type subline.
  // Slightly heavier weight so it reads as a control affordance rather than
  // typographic decoration.
  expandIcon: { fontSize: 10, fontWeight: '700' },
  // +1 / +4 quick-add buttons — primary (+1) is filled accent, secondary (+4) is outlined.
  qtyBtn: {
    minWidth: 36, height: 28, borderRadius: 8, borderWidth: 1,
    alignItems: 'center', justifyContent: 'center', paddingHorizontal: 6,
  },
  qtyBtnPrimary: { borderColor: 'transparent' },
  qtyBtnText: { fontSize: 13, fontWeight: '700' },
  qtyBtnTextPrimary: { color: '#fff', fontSize: 13, fontWeight: '700' },
});
