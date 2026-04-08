import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useSynergySearch } from '../api/hooks';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function FindSimilar({ card }: Props) {
  const router = useRouter();
  const { data: similar = [], isLoading } = useSynergySearch(card.name);

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Find Similar</Text>
      {isLoading ? (
        <ActivityIndicator color="#4ecdc4" />
      ) : similar.length === 0 ? (
        <Text style={styles.empty}>No similar cards found</Text>
      ) : (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.scroll}
        >
          {similar.slice(0, 10).map((c) => (
            <TouchableOpacity
              key={c.scryfall_id}
              style={styles.chip}
              onPress={() => router.push(`/card/${c.scryfall_id}`)}
            >
              <Text style={styles.chipText} numberOfLines={1}>{c.name}</Text>
              <Text style={styles.chipMeta}>{c.mana_cost}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { backgroundColor: '#0f0f1a', borderRadius: 8, padding: 12, marginTop: 12 },
  heading: { color: '#4ecdc4', fontWeight: '700', marginBottom: 8, fontSize: 13 },
  scroll: { gap: 8 },
  chip: { backgroundColor: '#1a1c23', borderRadius: 8, padding: 8, maxWidth: 120 },
  chipText: { color: '#fff', fontSize: 11, fontWeight: '600' },
  chipMeta: { color: '#888', fontSize: 10, marginTop: 2 },
  empty: { color: '#555', fontSize: 12 },
});
