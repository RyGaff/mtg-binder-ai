import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, Alert, Linking, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useSynergyFromCard } from '../api/hooks';
import { useStore } from '../store/useStore';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import { fetchCardByName } from '../api/scryfall';
import { upsertCard } from '../db/cards';
import type { CachedCard } from '../db/cards';
import { slugify, isCommanderEligible, type SynergyEntry } from '../api/edhrec';

type Props = { card: CachedCard };

export function Synergy({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const { data, isLoading, isError, fetchStatus, failureCount, refetch } = useSynergyFromCard(card);
  const entries = data?.entries ?? [];
  // Keep showing a spinner while retries are pending/in-flight. Only surface
  // "no data" once we have a successful fetch that returned empty.
  const retrying = fetchStatus === 'fetching' || (isError && failureCount < 5);
  const metric = data?.metric ?? 'synergy';
  const heading = metric === 'synergy' ? 'Synergies' : 'Inclusion';

  const openEntry = async (entry: SynergyEntry) => {
    if (entry.scryfall_id) {
      useStore.getState().markInternalTrailNav();
      router.replace(`/card/${entry.scryfall_id}`);
      return;
    }
    try {
      const resolved = await fetchCardByName(entry.name);
      upsertCard(resolved);
      useStore.getState().markInternalTrailNav();
      router.replace(`/card/${resolved.scryfall_id}`);
    } catch (err) {
      console.warn('[synergy] openEntry failed', entry.name, err);
      Alert.alert('Could not open card', `${entry.name}: ${String(err)}`);
    }
  };

  const edhrecUrl = `https://edhrec.com/${isCommanderEligible(card) ? 'commanders' : 'cards'}/${slugify(card.name)}`;

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <View style={styles.headerRow}>
        <Text style={[styles.heading, { color: theme.accent }]}>{heading}</Text>
        <TouchableOpacity onPress={() => Linking.openURL(edhrecUrl).catch((e) => Alert.alert('Could not open link', String(e)))} hitSlop={8}>
          <Text style={[styles.attribution, { color: theme.textSecondary }]}>
            Data from <Text style={{ color: theme.accent }}>EDHREC</Text>
          </Text>
        </TouchableOpacity>
      </View>

      {isLoading || (retrying && entries.length === 0) ? (
        <ActivityIndicator color={theme.accent} style={styles.loader} />
      ) : isError && entries.length === 0 ? (
        <TouchableOpacity onPress={() => refetch()} hitSlop={8}>
          <Text style={[styles.empty, { color: theme.textSecondary }]}>
            Could not load {heading.toLowerCase()}. Tap to retry.
          </Text>
        </TouchableOpacity>
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
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  heading: { fontWeight: '700', fontSize: 13 },
  attribution: { fontSize: 10 },
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
