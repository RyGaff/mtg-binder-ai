import { useEffect, useState } from 'react';
import {
  ActivityIndicator, FlatList, KeyboardAvoidingView, Modal,
  Pressable, StyleSheet, Text, TextInput, View,
} from 'react-native';
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
];

type Props = {
  visible: boolean;
  deckId: number;
  onClose: () => void;
};

export function AddCardsSheet({ visible, deckId, onClose }: Props) {
  const t = useTheme();
  const qc = useQueryClient();

  const [board, setBoard] = useState<Board>('main');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<CachedCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState<string[]>([]); // recent card names (toast strip)

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
    }, 300);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [query, visible]);

  const add = (card: CachedCard) => {
    upsertCard(card);
    addCardToDeck({ deck_id: deckId, scryfall_id: card.scryfall_id, quantity: 1, board });
    qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
    qc.invalidateQueries({ queryKey: ['decks'] });
    void ensureDeckArt(deckId, card.scryfall_id).then(() => {
      qc.invalidateQueries({ queryKey: ['deck', deckId] });
      qc.invalidateQueries({ queryKey: ['decks'] });
    });
    setRecent((r) => [card.name, ...r].slice(0, 3));
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable style={s.backdrop} onPress={onClose}>
        <KeyboardAvoidingView behavior="padding" style={s.anchor}>
          <Pressable onPress={(e) => e.stopPropagation()} style={[s.sheet, { backgroundColor: t.surface, borderColor: t.border }]}>
            <View style={s.handle} />

            <View style={s.titleRow}>
              <Text style={[s.title, { color: t.text }]}>Add to</Text>
              <View style={s.boardChips}>
                {BOARDS.map((b) => (
                  <Pressable key={b.key} onPress={() => setBoard(b.key)}
                    style={[s.boardChip, { backgroundColor: b.key === board ? t.accent : t.surfaceAlt }]}>
                    <Text style={[s.boardChipText, { color: b.key === board ? '#fff' : t.text }]}>{b.label}</Text>
                  </Pressable>
                ))}
              </View>
              <Pressable onPress={onClose} hitSlop={8}>
                <Text style={[s.close, { color: t.textSecondary }]}>✕</Text>
              </Pressable>
            </View>

            <TextInput
              value={query} onChangeText={setQuery}
              placeholder="Search cards…" placeholderTextColor={t.textSecondary}
              autoFocus autoCorrect={false} autoCapitalize="none"
              style={[s.input, { backgroundColor: t.bg, color: t.text, borderColor: t.border }]}
            />

            {recent.length > 0 ? (
              <Text style={[s.recent, { color: t.textSecondary }]} numberOfLines={1}>
                Added: {recent.join(' · ')}
              </Text>
            ) : null}

            {loading ? (
              <ActivityIndicator color={t.accent} style={{ paddingVertical: 24 }} />
            ) : query.trim().length < 2 ? (
              <Text style={[s.hint, { color: t.textSecondary }]}>Type at least 2 characters.</Text>
            ) : results.length === 0 ? (
              <Text style={[s.hint, { color: t.textSecondary }]}>No matches.</Text>
            ) : (
              <FlatList
                data={results}
                keyExtractor={(c) => c.scryfall_id}
                keyboardShouldPersistTaps="handled"
                style={s.results}
                renderItem={({ item }) => (
                  <Pressable onPress={() => add(item)} style={({ pressed }) => [s.resultRow, { borderBottomColor: t.border }, pressed && { opacity: 0.6 }]}>
                    <ManaCost cost={item.mana_cost} size={11} />
                    <Text style={[s.resultName, { color: t.text }]} numberOfLines={1}>{item.name}</Text>
                    <Text style={[s.plus, { color: t.accent }]}>+</Text>
                  </Pressable>
                )}
              />
            )}
          </Pressable>
        </KeyboardAvoidingView>
      </Pressable>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)' },
  anchor: { flex: 1, justifyContent: 'flex-end' },
  sheet: { borderTopLeftRadius: 20, borderTopRightRadius: 20, borderWidth: 1, padding: 16, gap: 10, maxHeight: '85%' },
  handle: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, backgroundColor: '#999', marginBottom: 4 },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 14, fontWeight: '700' },
  boardChips: { flex: 1, flexDirection: 'row', gap: 4 },
  boardChip: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  boardChipText: { fontSize: 11, fontWeight: '600' },
  close: { fontSize: 18, paddingHorizontal: 4 },
  input: { height: 42, borderRadius: 10, borderWidth: 1, paddingHorizontal: 12, fontSize: 14 },
  recent: { fontSize: 11 },
  hint: { fontSize: 13, textAlign: 'center', paddingVertical: 24 },
  results: { maxHeight: 380 },
  resultRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 10, paddingHorizontal: 4, borderBottomWidth: StyleSheet.hairlineWidth },
  resultName: { flex: 1, fontSize: 13, fontWeight: '600' },
  plus: { fontSize: 18, fontWeight: '700', paddingHorizontal: 4 },
});
