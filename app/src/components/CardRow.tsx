import { TouchableOpacity, View, Image, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function CardRow({ card }: Props) {
  const router = useRouter();
  const prices = JSON.parse(card.prices || '{}');

  return (
    <TouchableOpacity
      style={styles.row}
      onPress={() => router.push(`/card/${card.scryfall_id}`)}
    >
      {card.image_uri ? (
        <Image source={{ uri: card.image_uri }} style={styles.image} resizeMode="cover" />
      ) : (
        <View style={[styles.image, styles.imagePlaceholder]} />
      )}
      <View style={styles.info}>
        <Text style={styles.name}>{card.name}</Text>
        <Text style={styles.type}>
          {card.mana_cost}{'  '}{card.type_line}
        </Text>
        {prices.usd ? <Text style={styles.price}>${prices.usd}</Text> : null}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    gap: 10,
    padding: 10,
    backgroundColor: '#1a1c23',
    borderRadius: 8,
    marginBottom: 6,
  },
  image: { width: 40, height: 56, borderRadius: 4 },
  imagePlaceholder: { backgroundColor: '#2a1a3e' },
  info: { flex: 1, justifyContent: 'center' },
  name: { color: '#fff', fontSize: 14, fontWeight: '600' },
  type: { color: '#888', fontSize: 12, marginTop: 2 },
  price: { color: '#4ecdc4', fontSize: 12, marginTop: 2 },
});
