import { memo, useCallback, useMemo } from 'react';
import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

function CardRowImpl({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const prices = useMemo(() => {
    try { return JSON.parse(card.prices || '{}') as { usd?: string }; }
    catch { return {}; }
  }, [card.prices]);

  const navigate = useCallback(
    () => router.push(`/card/${card.scryfall_id}`),
    [router, card.scryfall_id],
  );

  return (
    <TouchableOpacity style={[styles.row, { backgroundColor: theme.surface }]} onPress={navigate}>
      {card.image_uri ? (
        <PressableCardImage card={card} style={styles.image} onPress={navigate} />
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

export const CardRow = memo(CardRowImpl, (prev, next) => prev.card === next.card);

const styles = StyleSheet.create({
  row: { flexDirection: 'row', gap: 10, padding: 10, borderRadius: 8, marginBottom: 8, minHeight: 64 },
  image: { width: 40, height: 56, borderRadius: 4 },
  info: { flex: 1, justifyContent: 'center' },
  name: { fontSize: 14, fontWeight: '600' },
  type: { fontSize: 12, marginTop: 2 },
  price: { fontSize: 12, marginTop: 2 },
});
