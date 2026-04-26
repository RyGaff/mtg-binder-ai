import { memo, useCallback } from 'react';
import { TouchableOpacity, View, StyleSheet, type StyleProp, type ViewStyle } from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard; style?: StyleProp<ViewStyle> };

const ASPECT = 488 / 680;

function CardTileImpl({ card, style }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const navigate = useCallback(
    () => router.push(`/card/${card.scryfall_id}`),
    [router, card.scryfall_id],
  );
  return (
    <TouchableOpacity onPress={navigate} style={[styles.tile, style]}>
      {card.image_uri ? (
        <PressableCardImage card={card} style={styles.image} onPress={navigate} resizeMode="cover" thumb />
      ) : (
        <View style={[styles.image, { backgroundColor: theme.surfaceAlt }]} />
      )}
    </TouchableOpacity>
  );
}

export const CardTile = memo(CardTileImpl, (prev, next) => prev.card === next.card && prev.style === next.style);
export { ASPECT as CARD_ASPECT };

const styles = StyleSheet.create({
  tile: { padding: 4 },
  image: { width: '100%', aspectRatio: ASPECT, borderRadius: 6 },
});
