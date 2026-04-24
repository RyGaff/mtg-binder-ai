import {
  View,
  Text,
  SectionList,
  TouchableOpacity,
  Alert,
  StyleSheet,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useMemo } from 'react';
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
import { Icon } from '../../src/components/icons/Icon';

const BOARDS = ['commander', 'main', 'side'] as const;

export default function DeckDetailScreen() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const deckId = Number(id);
  const router = useRouter();
  const qc = useQueryClient();

  const { data: decks = [] } = useQuery({ queryKey: ['decks'], queryFn: getDecks });
  const { data: cards = [] } = useQuery({
    queryKey: ['deck-cards', deckId],
    queryFn: () => getDeckCards(deckId),
  });

  const deck = useMemo(() => decks.find((d) => d.id === deckId), [decks, deckId]);

  const sections = useMemo(
    () => BOARDS
      .map((board) => ({
        title: board.charAt(0).toUpperCase() + board.slice(1),
        data: cards.filter((c) => c.board === board),
      }))
      .filter((s) => s.data.length > 0),
    [cards],
  );

  const handleExport = async () => {
    try {
      const text = exportDeckAsText(deckId);
      const path = `${FileSystem.cacheDirectory}${deck?.name ?? 'deck'}.txt`;
      await FileSystem.writeAsStringAsync(path, text, { encoding: FileSystem.EncodingType.UTF8 });
      await Sharing.shareAsync(path, { mimeType: 'text/plain' });
    } catch {
      Alert.alert('Export Failed', 'Could not export deck.');
    }
  };

  const handleRemoveCard = useCallback(
    (card: DeckCard) => {
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
    },
    [deckId, qc],
  );

  const keyExtractor = useCallback((item: DeckCard) => `${item.scryfall_id}-${item.board}`, []);

  const renderSectionHeader = useCallback(
    ({ section }: { section: { title: string; data: DeckCard[] } }) => (
      <Text style={[styles.sectionHeader, { color: theme.accent }]}>
        {section.title} ({section.data.reduce((s, c) => s + c.quantity, 0)})
      </Text>
    ),
    [theme.accent],
  );

  const renderItem = useCallback(
    ({ item }: { item: DeckCard }) => (
      <TouchableOpacity
        style={[styles.cardRow, { borderBottomColor: theme.surface }]}
        onPress={() => router.push(`/card/${item.scryfall_id}`)}
        onLongPress={() => handleRemoveCard(item)}
        accessibilityRole="button"
        accessibilityLabel={`${item.quantity} ${item.name}`}
        accessibilityHint="Long press to remove"
      >
        <Text style={[styles.qty, { color: theme.textSecondary }]}>{item.quantity}×</Text>
        <View style={styles.cardInfo}>
          <Text style={[styles.cardName, { color: theme.text }]} numberOfLines={1} ellipsizeMode="tail">{item.name}</Text>
          <Text style={[styles.cardType, { color: theme.textSecondary }]} numberOfLines={1} ellipsizeMode="tail">
            {item.mana_cost}{'  '}{item.type_line}
          </Text>
        </View>
      </TouchableOpacity>
    ),
    [theme.surface, theme.textSecondary, theme.text, router, handleRemoveCard],
  );

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
          accessibilityRole="button"
          accessibilityLabel="Add cards"
        >
          <Icon name="plus" size={16} color={theme.text} strokeWidth={2.5} />
          <Text style={[styles.btnText, { color: theme.text }]}>Add Cards</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.btn, { backgroundColor: theme.surfaceAlt }]} onPress={handleExport}>
          <Text style={[styles.btnText, { color: theme.text }]}>Export</Text>
        </TouchableOpacity>
      </View>

      <SectionList
        sections={sections}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.list}
        renderSectionHeader={renderSectionHeader}
        renderItem={renderItem}
        initialNumToRender={20}
        maxToRenderPerBatch={20}
        windowSize={9}
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
  actions: { flexDirection: 'row', gap: 12, paddingHorizontal: 16, paddingVertical: 12 },
  btn: {
    flex: 1,
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
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
    paddingVertical: 12,
    minHeight: 44,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  qty: { fontSize: 14, width: 28, textAlign: 'right' },
  cardInfo: { flex: 1 },
  cardName: { fontSize: 14 },
  cardType: { fontSize: 11, marginTop: 2 },
  empty: { textAlign: 'center', marginTop: 60 },
});
