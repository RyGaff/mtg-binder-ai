import {
  View,
  Text,
  SectionList,
  TouchableOpacity,
  Alert,
  StyleSheet,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import {
  getDeckCards,
  getDecks,
  removeCardFromDeck,
  exportDeckAsText,
  type DeckCard,
} from '../../src/db/decks';
import { useTheme } from '../../src/theme/useTheme';

export default function DeckDetailScreen() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const deckId = Number(id);
  const router = useRouter();
  const qc = useQueryClient();

  const { data: decks = [] } = useQuery({
    queryKey: ['decks'],
    queryFn: getDecks,
  });

  const { data: cards = [] } = useQuery({
    queryKey: ['deck-cards', deckId],
    queryFn: () => getDeckCards(deckId),
  });

  const deck = decks.find((d) => d.id === deckId);

  const sections = (['commander', 'main', 'side'] as const)
    .map((board) => ({
      title: board.charAt(0).toUpperCase() + board.slice(1),
      data: cards.filter((c) => c.board === board),
    }))
    .filter((s) => s.data.length > 0);

  const handleExport = async () => {
    try {
      const text = exportDeckAsText(deckId);
      const path = `${FileSystem.cacheDirectory}${deck?.name ?? 'deck'}.txt`;
      await FileSystem.writeAsStringAsync(path, text, {
        encoding: FileSystem.EncodingType.UTF8,
      });
      await Sharing.shareAsync(path, { mimeType: 'text/plain' });
    } catch {
      Alert.alert('Export Failed', 'Could not export deck.');
    }
  };

  const handleRemoveCard = (card: DeckCard) => {
    Alert.alert('Remove', `Remove ${card.name}?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Remove',
        style: 'destructive',
        onPress: () => {
          removeCardFromDeck(deckId, card.scryfall_id, card.board);
          qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
        },
      },
    ]);
  };

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <View style={[styles.header, { backgroundColor: theme.surface }]}>
        <Text style={[styles.deckName, { color: theme.text }]}>{deck?.name ?? 'Deck'}</Text>
        <Text style={[styles.deckMeta, { color: theme.textSecondary }]}>
          {deck?.format} · {cards.length} cards
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.btn, { backgroundColor: theme.accent }]}
          onPress={() => router.push('/search')}
        >
          <Text style={[styles.btnText, { color: theme.text }]}>+ Add Cards</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btn, { backgroundColor: theme.surfaceAlt }]}
          onPress={handleExport}
        >
          <Text style={[styles.btnText, { color: theme.text }]}>Export</Text>
        </TouchableOpacity>
      </View>

      <SectionList
        sections={sections}
        keyExtractor={(item) => `${item.scryfall_id}-${item.board}`}
        contentContainerStyle={styles.list}
        renderSectionHeader={({ section }) => (
          <Text style={[styles.sectionHeader, { color: theme.accent }]}>
            {section.title} (
            {section.data.reduce((s, c) => s + c.quantity, 0)})
          </Text>
        )}
        renderItem={({ item }: { item: DeckCard }) => (
          <TouchableOpacity
            style={[styles.cardRow, { borderBottomColor: theme.surface }]}
            onPress={() => router.push(`/card/${item.scryfall_id}`)}
            onLongPress={() => handleRemoveCard(item)}
          >
            <Text style={[styles.qty, { color: theme.textSecondary }]}>{item.quantity}×</Text>
            <View style={styles.cardInfo}>
              <Text style={[styles.cardName, { color: theme.text }]}>{item.name}</Text>
              <Text style={[styles.cardType, { color: theme.textSecondary }]}>
                {item.mana_cost}{'  '}{item.type_line}
              </Text>
            </View>
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <Text style={[styles.empty, { color: theme.textSecondary }]}>
            No cards yet. Tap "+ Add Cards" to search.
          </Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  header: { padding: 16 },
  deckName: { fontSize: 20, fontWeight: '700' },
  deckMeta: { fontSize: 13, marginTop: 2 },
  actions: { flexDirection: 'row', gap: 8, padding: 12 },
  btn: {
    flex: 1,
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  btnText: { fontWeight: '600' },
  list: { padding: 12 },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginVertical: 8,
  },
  cardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    borderBottomWidth: 1,
  },
  qty: { fontSize: 14, width: 28, textAlign: 'right' },
  cardInfo: { flex: 1 },
  cardName: { fontSize: 14 },
  cardType: { fontSize: 11, marginTop: 2 },
  empty: { textAlign: 'center', marginTop: 60 },
});
