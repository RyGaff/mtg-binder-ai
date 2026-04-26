import { Pressable, SectionList, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import {
  exportDeckAsText, getDeck, getDeckCards, removeCardFromDeck, setDeckArt,
  type DeckCard,
} from '../../src/db/decks';
import { fetchArtCrop } from '../../src/api/scryfall';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';
import { AddCardsSheet } from '../../src/components/AddCardsSheet';
import { useActionSheet } from '../../src/components/ActionSheet';
import { DeckHero } from '../../src/components/DeckHero';
import { DeckInfoStrip } from '../../src/components/DeckInfoStrip';
import { DeckStatsPanel } from '../../src/components/DeckStatsPanel';
import { ManaCost } from '../../src/components/ManaCost';
import { boardPrice } from '../../src/utils/deckStats';
import { buildSections, type RowSection } from '../../src/utils/deckSections';

export default function DeckDetailScreen() {
  const t = useTheme();
  const router = useRouter();
  const qc = useQueryClient();
  const { id } = useLocalSearchParams<{ id: string }>();
  const deckId = Number(id);
  const setActiveDeckId = useStore((s) => s.setActiveDeckId);

  const { data: deck } = useQuery({ queryKey: ['deck', deckId], queryFn: () => getDeck(deckId) });
  const { data: cards = [] } = useQuery({ queryKey: ['deck-cards', deckId], queryFn: () => getDeckCards(deckId) });

  const [statsOpen, setStatsOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const sheet = useActionSheet();

  const sections = useMemo(() => buildSections(cards), [cards]);
  const mainCount = useMemo(() => cards.filter((c) => c.board === 'main').reduce((s, c) => s + c.quantity, 0), [cards]);
  const sideCount = useMemo(() => cards.filter((c) => c.board === 'side').reduce((s, c) => s + c.quantity, 0), [cards]);
  const totalPrice = useMemo(() => boardPrice(cards), [cards]);
  const colorIdentity = useMemo(() => {
    const set = new Set<string>();
    for (const c of cards) {
      try { for (const k of JSON.parse(c.color_identity || '[]') as string[]) if ('WUBRG'.includes(k)) set.add(k); }
      catch { /* skip */ }
    }
    return Array.from(set);
  }, [cards]);

  const mainCommanderCards = useMemo(() => cards.filter((c) => c.board === 'main' || c.board === 'commander'), [cards]);
  const sideCards = useMemo(() => cards.filter((c) => c.board === 'side'), [cards]);

  const cardOptions = useCallback((card: DeckCard) => {
    sheet.show({
      title: card.name,
      actions: [
        { label: 'Set as deck art', onPress: async () => {
          try {
            const uri = await fetchArtCrop(card.scryfall_id);
            if (uri) {
              setDeckArt(deckId, uri);
              qc.invalidateQueries({ queryKey: ['deck', deckId] });
              qc.invalidateQueries({ queryKey: ['decks'] });
            }
          } catch (e) { console.warn('fetchArtCrop failed', e); }
        } },
        { label: 'Remove', destructive: true, onPress: () => {
          removeCardFromDeck(deckId, card.scryfall_id, card.board);
          qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
          qc.invalidateQueries({ queryKey: ['decks'] });
        } },
      ],
    });
  }, [deckId, qc, sheet]);

  const exportDeck = useCallback(async () => {
    try {
      const text = exportDeckAsText(deckId);
      const path = `${FileSystem.cacheDirectory}${deck?.name ?? 'deck'}.txt`;
      await FileSystem.writeAsStringAsync(path, text, { encoding: FileSystem.EncodingType.UTF8 });
      await Sharing.shareAsync(path, { mimeType: 'text/plain' });
    } catch (e) { console.warn('exportDeck failed', e); }
  }, [deckId, deck?.name]);

  const more = useCallback(() => {
    sheet.show({
      title: deck?.name ?? 'Deck',
      actions: [{ label: 'Export', onPress: () => { void exportDeck(); } }],
    });
  }, [deck?.name, exportDeck, sheet]);

  const openAdd = useCallback(() => {
    setActiveDeckId(deckId);
    setAddOpen(true);
  }, [deckId, setActiveDeckId]);

  const renderSectionHeader = useCallback(({ section }: { section: RowSection }) => {
    const isBoard = section.kind === 'board';
    return (
      <View style={[s.section, isBoard ? s.boardSection : null, isBoard ? { borderTopColor: t.border } : null]}>
        <Text style={[isBoard ? s.boardTitle : s.typeTitle, { color: isBoard ? t.text : t.textSecondary }]}>
          {section.title}
        </Text>
        <Text style={[s.sectionCount, { color: t.textSecondary }]}>
          {section.count}{section.price != null ? ` · $${section.price.toFixed(2)}` : ''}
        </Text>
      </View>
    );
  }, [t]);

  const renderItem = useCallback(({ item }: { item: DeckCard }) => (
    <Pressable
      onPress={() => router.push(`/card/${item.scryfall_id}`)}
      onLongPress={() => cardOptions(item)}
      style={({ pressed }) => [s.row, { borderBottomColor: t.border }, pressed && { opacity: 0.6 }]}
    >
      <Text style={[s.qty, { color: t.textSecondary }]}>{item.quantity}</Text>
      <ManaCost cost={item.mana_cost} size={12} />
      <Text style={[s.name, { color: t.text }]} numberOfLines={1}>{item.name}</Text>
    </Pressable>
  ), [router, cardOptions, t]);

  return (
    <View style={[s.screen, { backgroundColor: t.bg }]}>
      <DeckHero
        name={deck?.name ?? 'Deck'}
        artCropUri={deck?.art_crop_uri ?? ''}
        onBack={() => router.back()}
        onMore={more}
      />
      <DeckInfoStrip
        format={deck?.format ?? ''}
        colorIdentity={colorIdentity}
        mainCount={mainCount}
        sideCount={sideCount}
        totalPrice={totalPrice}
        expanded={statsOpen}
        onToggleStats={() => setStatsOpen((v) => !v)}
      />
      {statsOpen ? <DeckStatsPanel mainCommander={mainCommanderCards} side={sideCards} /> : null}

      <SectionList
        sections={sections}
        keyExtractor={(c, i) => `${c.board}:${c.scryfall_id}:${i}`}
        renderSectionHeader={renderSectionHeader}
        renderItem={renderItem}
        contentContainerStyle={s.list}
        ListEmptyComponent={
          <Text style={[s.empty, { color: t.textSecondary }]}>No cards yet. Tap + to add.</Text>
        }
      />

      <Pressable
        onPress={openAdd}
        style={({ pressed }) => [s.fab, { backgroundColor: t.accent }, pressed && { opacity: 0.8, transform: [{ scale: 0.96 }] }]}
        accessibilityRole="button"
        accessibilityLabel="Add cards"
      >
        <Text style={s.fabLabel}>+</Text>
      </Pressable>

      <AddCardsSheet visible={addOpen} deckId={deckId} onClose={() => setAddOpen(false)} />
      {sheet.node}
    </View>
  );
}

const s = StyleSheet.create({
  screen: { flex: 1 },
  list: { paddingBottom: 100 },
  section: { paddingHorizontal: 14, paddingTop: 10, paddingBottom: 4, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  boardSection: { paddingTop: 16, marginTop: 4, borderTopWidth: 1 },
  boardTitle: { fontSize: 11, fontWeight: '800', textTransform: 'uppercase', letterSpacing: 0.8 },
  typeTitle: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6 },
  sectionCount: { fontSize: 10, fontWeight: '600' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, paddingHorizontal: 14, borderBottomWidth: StyleSheet.hairlineWidth },
  qty: { width: 18, textAlign: 'right', fontSize: 11, fontWeight: '700' },
  name: { flex: 1, fontSize: 13, fontWeight: '600' },
  empty: { textAlign: 'center', marginTop: 60 },
  fab: {
    position: 'absolute', right: 24, bottom: 32, width: 56, height: 56, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center', elevation: 6,
    shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 6, shadowOffset: { width: 0, height: 3 },
  },
  fabLabel: { color: 'white', fontSize: 28, fontWeight: '300', lineHeight: 30 },
});
