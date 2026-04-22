import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, Alert, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useSynergyFromCard } from '../api/hooks';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import { fetchCardByName } from '../api/scryfall';
import { upsertCard } from '../db/cards';
import type { CachedCard } from '../db/cards';
import type { SynergyEntry } from '../api/edhrec';

type Props = { card: CachedCard };

export function Synergy({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const { data, isLoading, isError } = useSynergyFromCard(card);
  const entries = data?.entries ?? [];
  const metric = data?.metric ?? 'synergy';
  const heading = metric === 'synergy' ? 'Synergies' : 'Inclusion';

  const openEntry = async (entry: SynergyEntry) => {
    if (entry.scryfall_id) {
      router.push(`/card/${entry.scryfall_id}`);
      return;
    }
    try {
      const resolved = await fetchCardByName(entry.name);
      upsertCard(resolved);
      router.push(`/card/${resolved.scryfall_id}`);
    } catch (err) {
      console.warn('[synergy] openEntry failed', entry.name, err);
      Alert.alert('Could not open card', `${entry.name}: ${String(err)}`);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <Text style={[styles.heading, { color: theme.accent }]}>{heading}</Text>

      {isLoading ? (
        <ActivityIndicator color={theme.accent} style={styles.loader} />
      ) : isError ? (
        <Text style={[styles.empty, { color: theme.textSecondary }]}>Could not load {heading.toLowerCase()}</Text>
      ) : entries.length === 0 ? (
        <Text style={[styles.empty, { color: theme.textSecondary }]}>No EDHREC data for this card</Text>
      ) : (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.strip}
        >
          {entries.map((s) => (
            <TouchableOpacity
              key={s.name}
              style={styles.cardItem}
              onPress={() => openEntry(s)}
              accessibilityLabel={`${s.name}, ${metric} ${s.score} percent`}
            >
              <View>
                {s.image_uri ? (
                  <PressableCardImage
                    uri={s.image_uri}
                    style={[styles.cardImage, { backgroundColor: theme.surface }]}
                    onPress={() => openEntry(s)}
                  />
                ) : (
                  <View style={[styles.cardImage, { backgroundColor: theme.surfaceAlt }]} />
                )}
                <View style={[styles.badge, { backgroundColor: theme.accent }]}>
                  <Text style={[styles.badgeText, { color: theme.text }]}>{s.score}%</Text>
                </View>
              </View>
              <Text style={[styles.cardName, { color: theme.text }]} numberOfLines={1}>{s.name}</Text>
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
  loader: { marginVertical: 12 },
  strip: { gap: 8 },
  cardItem: { width: 100, alignItems: 'center' },
  cardImage: { width: 100, height: 140, borderRadius: 6 },
  badge: {
    position: 'absolute',
    top: 4,
    right: 4,
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  badgeText: { fontSize: 10, fontWeight: '700' },
  cardName: { fontSize: 10, fontWeight: '600', marginTop: 4, textAlign: 'center', width: 100 },
  empty: { fontSize: 12 },
});
