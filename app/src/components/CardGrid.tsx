import {
  FlatList,
  TouchableOpacity,
  View,
  Text,
  Image,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
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
              style={[styles.card, styles.addCard]}
              onPress={onAddPress}
            >
              <Text style={styles.addPlus}>+</Text>
              <Text style={styles.addLabel}>Add card</Text>
            </TouchableOpacity>
          );
        }

        const prices = JSON.parse(item.prices || '{}');
        const price = item.foil ? prices.usd_foil : prices.usd;

        return (
          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push(`/card/${item.scryfall_id}`)}
          >
            {item.image_uri ? (
              <Image
                source={{ uri: item.image_uri }}
                style={styles.image}
                resizeMode="cover"
              />
            ) : (
              <View style={[styles.image, styles.imagePlaceholder]}>
                <Text style={styles.placeholderText}>{item.name[0]}</Text>
              </View>
            )}
            <View style={styles.info}>
              <Text style={styles.name} numberOfLines={1}>
                {item.name}
              </Text>
              <Text style={styles.meta}>
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
  card: {
    width: CARD_WIDTH,
    backgroundColor: '#1a1c23',
    borderRadius: 8,
    overflow: 'hidden',
  },
  image: { width: CARD_WIDTH, height: CARD_WIDTH * 1.4 },
  imagePlaceholder: {
    backgroundColor: '#2a1a3e',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: { color: '#888', fontSize: 24 },
  info: { padding: 6 },
  name: { color: '#fff', fontSize: 11, fontWeight: '600' },
  meta: { color: '#888', fontSize: 10, marginTop: 2 },
  addCard: {
    borderWidth: 1,
    borderColor: '#4ecdc460',
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    height: CARD_WIDTH * 1.4 + 36,
  },
  addPlus: { color: '#4ecdc4', fontSize: 28, lineHeight: 32 },
  addLabel: { color: '#4ecdc4', fontSize: 11 },
});
