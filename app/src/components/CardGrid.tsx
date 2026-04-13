import {
  FlatList,
  TouchableOpacity,
  View,
  Text,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { CollectionEntryWithCard } from '../db/collection';

const COLS = 3;
const CARD_WIDTH =
  (Dimensions.get('window').width - 16 * 2 - 8 * (COLS - 1)) / COLS;

type AddButton = { isAddButton: true };
type GridItem = CollectionEntryWithCard | AddButton;

type Props = {
  entries: CollectionEntryWithCard[];
  onAddPress: () => void;
};

export function CardGrid({ entries, onAddPress }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const data: GridItem[] = [...entries, { isAddButton: true }];

  return (
    <FlatList
      data={data}
      numColumns={COLS}
      keyExtractor={(item) =>
        'isAddButton' in item ? 'add-btn' : String(item.id)
      }
      contentContainerStyle={styles.list}
      columnWrapperStyle={styles.row}
      renderItem={({ item }) => {
        if ('isAddButton' in item) {
          return (
            <TouchableOpacity
              style={[styles.card, styles.addCard, { borderColor: theme.accent + '60' }]}
              onPress={onAddPress}
            >
              <Text style={[styles.addPlus, { color: theme.accent }]}>+</Text>
              <Text style={[styles.addLabel, { color: theme.accent }]}>Add card</Text>
            </TouchableOpacity>
          );
        }

        const prices = JSON.parse(item.prices || '{}');
        const price = item.foil ? prices.usd_foil : prices.usd;

        return (
          <TouchableOpacity
            style={[styles.card, { backgroundColor: theme.surface }]}
            onPress={() => router.push(`/card/${item.scryfall_id}`)}
          >
            {item.image_uri ? (
              <PressableCardImage
                uri={item.image_uri}
                style={styles.image}
                onPress={() => router.push(`/card/${item.scryfall_id}`)}
              />
            ) : (
              <View style={[styles.image, { backgroundColor: theme.surfaceAlt, alignItems: 'center', justifyContent: 'center' }]}>
                <Text style={[styles.placeholderText, { color: theme.textSecondary }]}>{item.name[0]}</Text>
              </View>
            )}
            <View style={styles.info}>
              <Text style={[styles.name, { color: theme.text }]} numberOfLines={1}>
                {item.name}
              </Text>
              <Text style={[styles.meta, { color: theme.textSecondary }]}>
                ×{item.quantity}
                {item.foil ? ' ✨' : ''} · {price ? `$${price}` : '—'}
              </Text>
            </View>
          </TouchableOpacity>
        );
      }}
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: 16 },
  row: { gap: 8, marginBottom: 8 },
  card: { width: CARD_WIDTH, borderRadius: 8, overflow: 'hidden' },
  image: { width: CARD_WIDTH, height: CARD_WIDTH * 1.4 },
  placeholderText: { fontSize: 24 },
  info: { padding: 6 },
  name: { fontSize: 11, fontWeight: '600' },
  meta: { fontSize: 10, marginTop: 2 },
  addCard: {
    borderWidth: 1,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    height: CARD_WIDTH * 1.4 + 36,
  },
  addPlus: { fontSize: 28, lineHeight: 32 },
  addLabel: { fontSize: 11 },
});
