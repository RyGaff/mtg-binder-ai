import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function CardRow({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const prices = JSON.parse(card.prices || '{}');
  const navigate = () => router.push(`/card/${card.scryfall_id}`);

  return (
    <TouchableOpacity style={[styles.row, { backgroundColor: theme.surface }]} onPress={navigate}>
      {card.image_uri ? (
        <PressableCardImage uri={card.image_uri} style={styles.image} onPress={navigate} />
      ) : (
        <View style={[styles.image, { backgroundColor: theme.surfaceAlt }]} />
      )}
      <View style={styles.info}>
        <Text style={[styles.name, { color: theme.text }]}>{card.name}</Text>
        <Text style={[styles.type, { color: theme.textSecondary }]}>
          {card.mana_cost}{'  '}{card.type_line}
        </Text>
        {prices.usd ? <Text style={[styles.price, { color: theme.accent }]}>${prices.usd}</Text> : null}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', gap: 10, padding: 10, borderRadius: 8, marginBottom: 6 },
  image: { width: 40, height: 56, borderRadius: 4 },
  info: { flex: 1, justifyContent: 'center' },
  name: { fontSize: 14, fontWeight: '600' },
  type: { fontSize: 12, marginTop: 2 },
  price: { fontSize: 12, marginTop: 2 },
});
