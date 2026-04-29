import { useState } from 'react';
import { ActivityIndicator, FlatList, Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import { Image } from 'expo-image';
import { useQueryClient } from '@tanstack/react-query';
import { usePrintings } from '../api/hooks';
import { useTheme } from '../theme/useTheme';
import { fetchCardById } from '../api/scryfall';
import { upsertCard } from '../db/cards';
import { changePrintingInDeck, type Board, type DeckCard } from '../db/decks';
import type { PrintingSummary } from '../api/scryfall';

type Props = {
  visible: boolean;
  card: DeckCard | null;
  deckId: number;
  onClose: () => void;
};

/**
 * Bottom sheet that lists all printings for the given deck card and swaps the
 * deck row's scryfall_id when one is picked. Fetches the chosen printing's
 * full card payload before swap so the cards-table FK is satisfied.
 */
export function ChangePrintingSheet({ visible, card, deckId, onClose }: Props) {
  const t = useTheme();
  const qc = useQueryClient();
  // usePrintings expects a CachedCard-shaped object; DeckCard is a superset
  // (extends CachedCard), so it's safe to pass directly.
  const { data: printings = [], isLoading, isError } = usePrintings(card ?? ({ name: '' } as DeckCard));
  const [busyId, setBusyId] = useState<string | null>(null);

  if (!card) return null;

  const choose = async (p: PrintingSummary) => {
    if (p.scryfall_id === card.scryfall_id) { onClose(); return; }
    setBusyId(p.scryfall_id);
    try {
      const fresh = await fetchCardById(p.scryfall_id);
      upsertCard(fresh);
      changePrintingInDeck(deckId, card.scryfall_id, p.scryfall_id, card.board as Board);
      qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
      qc.invalidateQueries({ queryKey: ['deck', deckId] });
      qc.invalidateQueries({ queryKey: ['decks'] });
      onClose();
    } catch (e) {
      console.warn('changePrinting failed', e);
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable style={s.backdrop} onPress={onClose}>
        <Pressable
          onPress={(e) => e.stopPropagation()}
          style={[s.card, { backgroundColor: t.surface, borderColor: t.border }]}
        >
          <Text style={[s.title, { color: t.text }]}>Change printing</Text>
          <Text style={[s.subtitle, { color: t.textSecondary }]} numberOfLines={1}>{card.name}</Text>
          {isLoading ? (
            <ActivityIndicator color={t.accent} style={s.loader} />
          ) : isError || printings.length === 0 ? (
            <Text style={[s.empty, { color: t.textSecondary }]}>No printings found.</Text>
          ) : (
            <FlatList
              data={printings}
              keyExtractor={(p) => p.scryfall_id}
              keyboardShouldPersistTaps="handled"
              style={s.list}
              contentContainerStyle={s.listContent}
              renderItem={({ item }) => {
                const isCurrent = item.scryfall_id === card.scryfall_id;
                const busy = busyId === item.scryfall_id;
                const thumb = item.image_uri ? item.image_uri.replace('/normal.', '/small.') : null;
                return (
                  <Pressable
                    onPress={() => { void choose(item); }}
                    disabled={busy}
                    style={({ pressed }) => [
                      s.row,
                      { borderBottomColor: t.border, backgroundColor: isCurrent ? t.surfaceAlt : 'transparent' },
                      pressed && { opacity: 0.6 },
                    ]}
                  >
                    {thumb ? (
                      <Image source={thumb} style={s.thumb} contentFit="cover" cachePolicy="memory-disk" recyclingKey={thumb} />
                    ) : (
                      <View style={[s.thumb, { backgroundColor: t.border }]} />
                    )}
                    <View style={s.info}>
                      <View style={s.topRow}>
                        <View style={[s.setSquare, { backgroundColor: t.accent }]}>
                          <Text style={[s.setCode, { color: t.text }]}>{item.set_code.toUpperCase()}</Text>
                        </View>
                        <Text style={[s.setName, { color: t.textSecondary }]} numberOfLines={1}>{item.set_name}</Text>
                        <Text style={[s.collector, { color: t.textSecondary }]}>#{item.collector_number}</Text>
                      </View>
                      <Text style={[s.price, { color: t.text }]}>
                        {item.prices.usd ? `$${item.prices.usd}` : '—'}
                      </Text>
                    </View>
                    {busy ? <ActivityIndicator color={t.accent} /> : isCurrent ? (
                      <Text style={[s.currentTag, { color: t.accent }]}>CURRENT</Text>
                    ) : null}
                  </Pressable>
                );
              }}
            />
          )}
          <Pressable
            onPress={onClose}
            style={({ pressed }) => [s.cancel, { borderColor: t.border, backgroundColor: t.surface }, pressed && { opacity: 0.75 }]}
          >
            <Text style={[s.cancelText, { color: t.text }]}>Cancel</Text>
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'flex-end' },
  card: { borderTopLeftRadius: 20, borderTopRightRadius: 20, borderWidth: 1, padding: 18, gap: 6, maxHeight: '80%' },
  title: { fontSize: 16, fontWeight: '700' },
  subtitle: { fontSize: 13, marginBottom: 4 },
  loader: { marginVertical: 24 },
  empty: { fontSize: 12, paddingVertical: 14, textAlign: 'center' },
  list: { flexGrow: 0 },
  listContent: { paddingBottom: 4 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, paddingHorizontal: 6, borderBottomWidth: StyleSheet.hairlineWidth, borderRadius: 6 },
  thumb: { width: 44, height: 62, borderRadius: 4 },
  info: { flex: 1, gap: 4 },
  topRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  setSquare: { width: 36, height: 20, borderRadius: 3, alignItems: 'center', justifyContent: 'center' },
  setCode: { fontSize: 9, fontWeight: '700' },
  setName: { flex: 1, fontSize: 11 },
  collector: { fontSize: 11 },
  price: { fontSize: 12, fontWeight: '600' },
  currentTag: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6 },
  cancel: { borderRadius: 10, paddingVertical: 14, alignItems: 'center', marginTop: 8, borderWidth: 1 },
  cancelText: { fontWeight: '700', fontSize: 14 },
});
