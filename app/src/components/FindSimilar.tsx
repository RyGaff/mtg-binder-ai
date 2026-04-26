import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, StyleSheet, useWindowDimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { useCallback, useMemo } from 'react';
import { useSimilarSearch } from '../api/hooks';
import { useStore } from '../store/useStore';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import { spacing, radius, font } from '../theme/themes';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function FindSimilar({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const embeddingStatus = useStore((s) => s.embeddingStatus);
  const { data: similar = [], isLoading, isError } = useSimilarSearch(card);
  const { width: winW } = useWindowDimensions();
  // Fit ~3.5 tiles into the visible row; clamp so they're never absurdly small or large.
  const tileSize = useMemo(() => {
    const w = Math.min(Math.max((winW - spacing.lg * 2) / 3.5, 84), 130);
    return { width: w, imageHeight: w * 1.4 };
  }, [winW]);

  const openCard = useCallback((scryfallId: string) => {
    useStore.getState().markInternalTrailNav();
    router.replace(`/card/${scryfallId}`);
  }, [router]);

  function renderBody() {
    if (embeddingStatus === 'downloading') {
      return (
        <View style={styles.downloadingRow}>
          <ActivityIndicator color={theme.accent} size="small" />
          <Text style={[styles.downloadingText, { color: theme.textSecondary }]}>Downloading embeddings...</Text>
        </View>
      );
    }
    if (isLoading) return <ActivityIndicator color={theme.accent} style={styles.loader} />;
    if (isError) return <Text style={[styles.empty, { color: theme.textSecondary }]}>Could not load similar cards</Text>;
    if (similar.length === 0) return <Text style={[styles.empty, { color: theme.textSecondary }]}>No similar cards found</Text>;

    const imgStyle = { width: tileSize.width, height: tileSize.imageHeight };
    return (
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.strip}>
        {similar.map(c => (
          <TouchableOpacity
            key={c.scryfall_id}
            style={[styles.cardItem, { width: tileSize.width }]}
            onPress={() => openCard(c.scryfall_id)}
            accessibilityRole="button"
            accessibilityLabel={c.name}
          >
            {c.image_uri ? (
              <PressableCardImage
                card={c}
                style={[styles.cardImage, imgStyle, { backgroundColor: theme.surface }]}
                onPress={() => openCard(c.scryfall_id)}
                thumb
              />
            ) : (
              <View style={[styles.cardImage, imgStyle, { backgroundColor: theme.surfaceAlt }]} />
            )}
            <Text style={[styles.cardName, { color: theme.text, width: tileSize.width }]} numberOfLines={1} ellipsizeMode="tail">{c.name}</Text>
            <Text style={[styles.cardMana, { color: theme.textSecondary }]} numberOfLines={1}>{c.mana_cost}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <Text style={[styles.heading, { color: theme.accent }]}>Similar Cards</Text>
      {renderBody()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { borderRadius: radius.md, padding: spacing.md, marginTop: spacing.md },
  heading: { fontWeight: '700', marginBottom: spacing.sm, fontSize: 13 },
  downloadingRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginTop: spacing.xs },
  downloadingText: { fontSize: font.small },
  loader: { marginVertical: spacing.md },
  strip: { gap: spacing.sm },
  cardItem: { alignItems: 'center' },
  cardImage: { borderRadius: radius.sm + 2 },
  cardName: { fontSize: 10, fontWeight: '600', marginTop: spacing.xs, textAlign: 'center' },
  cardMana: { fontSize: 10, marginTop: 1, textAlign: 'center' },
  empty: { fontSize: font.small },
});
