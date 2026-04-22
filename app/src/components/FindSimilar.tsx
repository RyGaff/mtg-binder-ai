import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useSimilarSearch } from '../api/hooks';
import { useStore } from '../store/useStore';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function FindSimilar({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const embeddingStatus = useStore((s) => s.embeddingStatus);
  const { data: similar = [], isLoading, isError } = useSimilarSearch(card);

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <Text style={[styles.heading, { color: theme.accent }]}>Similar Cards</Text>

      {embeddingStatus === 'downloading' ? (
        <View style={styles.downloadingRow}>
          <ActivityIndicator color={theme.accent} size="small" />
          <Text style={[styles.downloadingText, { color: theme.textSecondary }]}>Downloading embeddings...</Text>
        </View>
      ) : isLoading ? (
        <ActivityIndicator color={theme.accent} style={styles.loader} />
      ) : isError ? (
        <Text style={[styles.empty, { color: theme.textSecondary }]}>Could not load similar cards</Text>
      ) : similar.length === 0 ? (
        <Text style={[styles.empty, { color: theme.textSecondary }]}>No similar cards found</Text>
      ) : (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.strip}
        >
          {similar.map(c => (
            <TouchableOpacity
              key={c.scryfall_id}
              style={styles.cardItem}
              onPress={() => router.push(`/card/${c.scryfall_id}`)}
              accessibilityLabel={c.name}
            >
              {c.image_uri ? (
                <PressableCardImage
                  uri={c.image_uri}
                  style={[styles.cardImage, { backgroundColor: theme.surface }]}
                  onPress={() => router.push(`/card/${c.scryfall_id}`)}
                />
              ) : (
                <View style={[styles.cardImage, { backgroundColor: theme.surfaceAlt }]} />
              )}
              <Text style={[styles.cardName, { color: theme.text }]} numberOfLines={1}>{c.name}</Text>
              <Text style={[styles.cardMana, { color: theme.textSecondary }]} numberOfLines={1}>{c.mana_cost}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { borderRadius: 8, padding: 12, marginTop: 12 },
  heading: { fontWeight: '700', marginBottom: 8, fontSize: 13 },
  downloadingRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 },
  downloadingText: { fontSize: 12 },
  loader: { marginVertical: 12 },
  strip: { gap: 8 },
  cardItem: { width: 100, alignItems: 'center' },
  cardImage: { width: 100, height: 140, borderRadius: 6 },
  cardName: { fontSize: 10, fontWeight: '600', marginTop: 4, textAlign: 'center', width: 100 },
  cardMana: { fontSize: 10, marginTop: 1, textAlign: 'center' },
  empty: { fontSize: 12 },
});
