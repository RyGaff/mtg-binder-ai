import { memo, useCallback, useMemo } from 'react';
import {
  FlatList, TouchableOpacity, View, Text, StyleSheet, Dimensions,
  type ListRenderItem,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import { Icon } from './icons/Icon';
import type { CollectionEntryWithCard } from '../db/collection';

const COLS = 3;
const CARD_WIDTH = (Dimensions.get('window').width - 16 * 2 - 8 * (COLS - 1)) / COLS;
const CARD_IMAGE_HEIGHT = CARD_WIDTH * 1.4;
const ROW_HEIGHT = CARD_IMAGE_HEIGHT + 36 + 8;

type AddButton = { isAddButton: true };
type GridItem = CollectionEntryWithCard | AddButton;
type Props = { entries: CollectionEntryWithCard[]; onAddPress: () => void };
type Router = ReturnType<typeof useRouter>;

function keyExtractor(item: GridItem) {
  return 'isAddButton' in item ? 'add-btn' : String(item.id);
}

function getItemLayout(_: ArrayLike<GridItem> | null | undefined, index: number) {
  return { length: ROW_HEIGHT, offset: ROW_HEIGHT * Math.floor(index / COLS), index };
}

const AddCard = memo(function AddCard({ onPress }: { onPress: () => void }) {
  const theme = useTheme();
  return (
    <TouchableOpacity
      style={[styles.card, styles.addCard, { borderColor: theme.accent + '60' }]}
      onPress={onPress}
    >
      <Icon name="plus" size={28} color={theme.accent} strokeWidth={2.5} />
      <Text style={[styles.addLabel, { color: theme.accent }]}>Add card</Text>
    </TouchableOpacity>
  );
});

const CollectionCard = memo(
  function CollectionCard({ entry, router }: { entry: CollectionEntryWithCard; router: Router }) {
    const theme = useTheme();
    const price = useMemo(() => {
      let prices: { usd?: string; usd_foil?: string } = {};
      try { prices = JSON.parse(entry.prices || '{}'); } catch { prices = {}; }
      return entry.foil ? prices.usd_foil : prices.usd;
    }, [entry.prices, entry.foil]);

    const navigate = useCallback(
      () => router.push(`/card/${entry.scryfall_id}`),
      [router, entry.scryfall_id],
    );

    return (
      <TouchableOpacity style={[styles.card, { backgroundColor: theme.surface }]} onPress={navigate}>
        {entry.image_uri ? (
          <PressableCardImage card={entry} style={styles.image} onPress={navigate} />
        ) : (
          <View style={[styles.image, styles.placeholder, { backgroundColor: theme.surfaceAlt }]}>
            <Text style={[styles.placeholderText, { color: theme.textSecondary }]}>{entry.name[0]}</Text>
          </View>
        )}
        <View style={styles.info}>
          <Text style={[styles.name, { color: theme.text }]} numberOfLines={1}>{entry.name}</Text>
          <View style={styles.metaRow}>
            <Text style={[styles.meta, { color: theme.textSecondary }]}>×{entry.quantity}</Text>
            {entry.foil && <Icon name="sparkle" size={9} color={theme.foilAccent} />}
            <Text style={[styles.meta, { color: theme.textSecondary }]}>
              · {price ? `$${price}` : '—'}
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  },
  (prev, next) => prev.entry === next.entry && prev.router === next.router,
);

export function CardGrid({ entries, onAddPress }: Props) {
  const router = useRouter();
  const data = useMemo<GridItem[]>(() => [...entries, { isAddButton: true }], [entries]);

  const renderItem = useCallback<ListRenderItem<GridItem>>(
    ({ item }) => 'isAddButton' in item
      ? <AddCard onPress={onAddPress} />
      : <CollectionCard entry={item} router={router} />,
    [router, onAddPress],
  );

  return (
    <FlatList
      data={data}
      numColumns={COLS}
      keyExtractor={keyExtractor}
      contentContainerStyle={styles.list}
      columnWrapperStyle={styles.row}
      renderItem={renderItem}
      getItemLayout={getItemLayout}
      initialNumToRender={12}
      maxToRenderPerBatch={6}
      windowSize={3}
      removeClippedSubviews={true}
      updateCellsBatchingPeriod={50}
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: 16 },
  row: { gap: 8, marginBottom: 8 },
  card: { width: CARD_WIDTH, borderRadius: 8, overflow: 'hidden' },
  image: { width: CARD_WIDTH, height: CARD_IMAGE_HEIGHT },
  placeholder: { alignItems: 'center', justifyContent: 'center' },
  placeholderText: { fontSize: 24 },
  info: { padding: 6 },
  name: { fontSize: 11, fontWeight: '600' },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 3, marginTop: 2 },
  meta: { fontSize: 10 },
  addCard: {
    borderWidth: 1, borderStyle: 'dashed',
    alignItems: 'center', justifyContent: 'center',
    gap: 4, height: CARD_IMAGE_HEIGHT + 36,
  },
  addLabel: { fontSize: 11 },
});
