import { useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/useTheme';
import { useStore } from '../store/useStore';
import { spacing, radius, font, MIN_TOUCH } from '../theme/themes';
import { fetchCardById } from '../api/scryfall';
import { getCardById, upsertCard } from '../db/cards';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };
type Part = { id: string; component: 'meld_part' | 'meld_result'; name: string };

function parseParts(raw: string | null | undefined): Part[] {
  if (!raw) return [];
  try {
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function MeldLinks({ card }: Props) {
  const theme = useTheme();
  const router = useRouter();
  const open = useCallback(async (id: string, name: string) => {
    try {
      if (!getCardById(id)) upsertCard(await fetchCardById(id));
      useStore.getState().markInternalTrailNav();
      router.replace(`/card/${id}`);
    } catch (err) {
      console.warn('[meld] openCard failed', name, err);
    }
  }, [router]);

  const parts = parseParts(card.all_parts);
  if (parts.length === 0) return null;
  const isResult = parts.find((p) => p.id === card.scryfall_id)?.component === 'meld_result';
  const others = parts.filter((p) => p.id !== card.scryfall_id);
  if (others.length === 0) return null;

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <Text style={[styles.heading, { color: theme.accent }]}>{isResult ? 'Melds from' : 'Melds into'}</Text>
      <View style={styles.row}>
        {others.map((p, i) => (
          <View key={p.id} style={styles.row}>
            {i > 0 && <Text style={[styles.plus, { color: theme.textSecondary }]}> + </Text>}
            <TouchableOpacity
              onPress={() => open(p.id, p.name)}
              style={styles.link}
              hitSlop={8}
              accessibilityRole="button"
              accessibilityLabel={`Open ${p.name}`}
            >
              <Text style={[styles.linkText, { color: theme.accent }]}>{p.name}</Text>
            </TouchableOpacity>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginTop: spacing.lg, padding: spacing.md, borderRadius: radius.md, gap: spacing.sm },
  heading: { fontSize: font.caption, textTransform: 'uppercase', letterSpacing: 1, fontWeight: '700' },
  row: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap' },
  link: { minHeight: MIN_TOUCH, justifyContent: 'center' },
  linkText: { fontSize: font.body, fontWeight: '600' },
  plus: { fontSize: font.body },
});
