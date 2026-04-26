import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, FlatList, KeyboardAvoidingView, Modal, Platform,
  Pressable, StyleSheet, Text, TextInput, TouchableOpacity, View,
} from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useQueryClient } from '@tanstack/react-query';
import { addCardToDeck, ensureDeckArt, type Board } from '../db/decks';
import { upsertCard, type CachedCard } from '../db/cards';
import { searchScryfall } from '../api/scryfall';
import { useTheme } from '../theme/useTheme';
import { ManaCost } from './ManaCost';

const BOARDS: { key: Board; label: string }[] = [
  { key: 'main', label: 'Main' },
  { key: 'side', label: 'Side' },
  { key: 'commander', label: 'Cmdr' },
  { key: 'considering', label: 'Maybe' },
];

type Props = {
  visible: boolean;
  deckId: number;
  onClose: () => void;
  /** Called right before the sheet closes to navigate to a card detail. The parent
      uses this to set a "resume on focus" flag so the sheet auto-reopens with the
      same search when the user returns from the detail screen. */
  onInspect?: () => void;
};

export function AddCardsSheet({ visible, deckId, onClose, onInspect }: Props) {
  const t = useTheme();
  const qc = useQueryClient();
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

  // Search-as-you-type — 150ms debounce instead of 300ms. Feels closer to instant
  // without stampeding Scryfall. Aborts in-flight on every keystroke.
  useEffect(() => {
    if (!visible) return;
    const q = query.trim();
    if (q.length < 2) { setResults([]); setLoading(false); return; }
    const ctrl = new AbortController();
    setLoading(true);
    const timer = setTimeout(() => {
      searchScryfall(q, 1, ctrl.signal)
        .then((r) => { setResults(r); setLoading(false); })
        .catch(() => { if (!ctrl.signal.aborted) setLoading(false); });
    }, 150);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [query, visible]);

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

  const add = (card: CachedCard) => {
    upsertCard(card);
    addCardToDeck({ deck_id: deckId, scryfall_id: card.scryfall_id, quantity: 1, board });
    qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
    qc.invalidateQueries({ queryKey: ['decks'] });
    void ensureDeckArt(deckId, card.scryfall_id).then(() => {
      qc.invalidateQueries({ queryKey: ['deck', deckId] });
      qc.invalidateQueries({ queryKey: ['decks'] });
    });
    setRecent((r) => {
      const next = [card, ...r.filter((c) => c.scryfall_id !== card.scryfall_id)];
      return next.slice(0, 6);
    });
  };

  // Submit (return key) adds the top result — the rapid-add power user move.
  const onSubmit = () => {
    if (results.length > 0) add(results[0]);
  };

  // Tapping a result's main row navigates to the card detail page (same inspect
  // experience as every other card surface in the app). The sheet closes first so
  // the detail screen isn't covered by the modal backdrop. onInspect tells the parent
  // to flag a "resume on focus" so returning to the deck reopens the sheet with the
  // same query/results retained (we don't reset state on close).
  const inspect = (card: CachedCard) => {
    upsertCard(card);
    onInspect?.();
    onClose();
    router.push(`/card/${card.scryfall_id}`);
  };

  // Pre-build per-result thumbnail URI (small variant).
  const thumbFor = (c: CachedCard): string | null =>
    c.image_uri ? c.image_uri.replace('/normal.', '/small.') : null;

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
          {/* Card stops bubble propagation so taps inside don't dismiss. */}
          <Pressable
            onPress={(e) => e.stopPropagation()}
            style={[s.card, { backgroundColor: t.surface, borderColor: t.border }]}
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
                onSubmitEditing={onSubmit}
                placeholder="Search Scryfall…"
                placeholderTextColor={t.textSecondary}
                returnKeyType="go"
                blurOnSubmit={false}
                autoCorrect={false}
                autoCapitalize="none"
                style={[s.input, { color: t.text }]}
              />
              {query.length > 0 ? (
                <Pressable onPress={() => setQuery('')} hitSlop={8} style={s.clear}>
                  <Text style={[s.clearIcon, { color: t.textSecondary }]}>×</Text>
                </Pressable>
              ) : null}
            </View>

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
                <ActivityIndicator color={t.accent} style={{ paddingVertical: 24 }} />
              ) : query.trim().length < 2 ? (
                <Text style={[s.hint, { color: t.textSecondary }]}>Type 2+ characters. Press return to add the top result.</Text>
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
                  renderItem={({ item, index }) => {
                    const thumb = thumbFor(item);
                    return (
                      <View style={s.resultRow}>
                        <TouchableOpacity
                          onPress={() => inspect(item)}
                          style={s.resultMain}
                          accessibilityLabel={`Inspect ${item.name}`}
                        >
                          {thumb ? (
                            <Image source={thumb} style={s.resultThumb} contentFit="cover" cachePolicy="memory-disk" recyclingKey={thumb} />
                          ) : (
                            <View style={[s.resultThumb, { backgroundColor: t.surfaceAlt }]} />
                          )}
                          <View style={s.resultMeta}>
                            <View style={s.resultTopRow}>
                              <Text style={[s.resultName, { color: t.text }]} numberOfLines={1}>{item.name}</Text>
                              {index === 0 ? (
                                <Text style={[s.returnHint, { color: t.textSecondary }]}>↵</Text>
                              ) : null}
                            </View>
                            <View style={s.resultSubRow}>
                              <ManaCost cost={item.mana_cost} size={11} />
                              <Text style={[s.resultType, { color: t.textSecondary }]} numberOfLines={1}>
                                {item.type_line ?? ''}
                              </Text>
                            </View>
                          </View>
                        </TouchableOpacity>
                        {/* +1 / +4 quick-add buttons stay distinct from row tap so power users
                            can build a deck fast without going to the row's main area. */}
                        <TouchableOpacity
                          onPress={() => { add(item); add(item); add(item); add(item); }}
                          hitSlop={6}
                          style={[s.qtyBtn, { borderColor: t.border }]}
                          accessibilityLabel={`Add 4x ${item.name}`}
                        >
                          <Text style={[s.qtyBtnText, { color: t.text }]}>+4</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                          onPress={() => add(item)}
                          hitSlop={6}
                          style={[s.qtyBtn, s.qtyBtnPrimary, { backgroundColor: t.accent }]}
                          accessibilityLabel={`Add 1x ${item.name}`}
                        >
                          <Text style={s.qtyBtnTextPrimary}>+1</Text>
                        </TouchableOpacity>
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

  // ---- Search input ----
  inputWrap: {
    flexDirection: 'row', alignItems: 'center',
    height: 48, borderRadius: 12, borderWidth: 1, paddingHorizontal: 12, gap: 8,
  },
  inputIcon: { fontSize: 18 },
  input: { flex: 1, fontSize: 16, paddingVertical: 0 },
  clear: { width: 24, height: 24, alignItems: 'center', justifyContent: 'center' },
  clearIcon: { fontSize: 20, fontWeight: '700' },

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
  resultMain: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 10 },
  // 30×42 thumbnail — shrunk from 36×50 so rows are shorter; still readable.
  resultThumb: { width: 30, height: 42, borderRadius: 4 },
  resultMeta: { flex: 1, gap: 2 },
  resultTopRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  resultName: { flex: 1, fontSize: 14, fontWeight: '600' },
  returnHint: { fontSize: 12, fontWeight: '700' },
  resultSubRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  resultType: { flex: 1, fontSize: 11 },
  // +1 / +4 quick-add buttons — primary (+1) is filled accent, secondary (+4) is outlined.
  qtyBtn: {
    minWidth: 36, height: 28, borderRadius: 8, borderWidth: 1,
    alignItems: 'center', justifyContent: 'center', paddingHorizontal: 6,
  },
  qtyBtnPrimary: { borderColor: 'transparent' },
  qtyBtnText: { fontSize: 13, fontWeight: '700' },
  qtyBtnTextPrimary: { color: '#fff', fontSize: 13, fontWeight: '700' },
});
