import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useStore } from '../src/store/useStore';
import { useTheme } from '../src/theme/useTheme';
import { getCollectionTotalValue, getFoilCount, getTotalCardCount } from '../src/db/collection';
import type { ThemeName } from '../src/theme/themes';

const BUILT_IN_THEMES: { name: ThemeName; label: string }[] = [
  { name: 'dark', label: 'Dark' },
  { name: 'light', label: 'Light' },
  { name: 'amoled', label: 'AMOLED' },
];

const FEEDBACK_EMAIL = 'lotusfieldmtg@gmail.com';

export default function ProfileScreen() {
  const router = useRouter();
  const { theme, setTheme, customThemes } = useStore();
  const t = useTheme();

  const { data: totalValue = 0 } = useQuery({
    queryKey: ['collection-value'],
    queryFn: getCollectionTotalValue,
  });

  const { data: totalCards = 0 } = useQuery({
    queryKey: ['total-cards'],
    queryFn: getTotalCardCount,
  });

  const { data: foilCount = 0 } = useQuery({
    queryKey: ['foil-count'],
    queryFn: getFoilCount,
  });

  return (
    <View style={[styles.screen, { backgroundColor: t.bg }]}>
      <TouchableOpacity
        style={styles.closeBtn}
        onPress={() => (router.canGoBack() ? router.back() : router.replace('/'))}
        accessibilityLabel="Close"
        accessibilityRole="button"
      >
        <Text style={[styles.closeBtnText, { color: t.textSecondary }]}>✕</Text>
      </TouchableOpacity>

      <Text style={[styles.screenTitle, { color: t.text }]}>Profile</Text>

      {/* Stats */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Collection</Text>
        <View style={[styles.statsCard, { backgroundColor: t.surface }]}>
          <View style={styles.statItem}>
            <Text style={[styles.statValue, { color: t.accent }]}>{totalCards}</Text>
            <Text style={[styles.statLabel, { color: t.textSecondary }]}>Cards</Text>
          </View>
          <View style={[styles.statDivider, { backgroundColor: t.border }]} />
          <View style={styles.statItem}>
            <Text style={[styles.statValue, { color: t.accent }]}>${totalValue.toFixed(2)}</Text>
            <Text style={[styles.statLabel, { color: t.textSecondary }]}>Value</Text>
          </View>
          <View style={[styles.statDivider, { backgroundColor: t.border }]} />
          <View style={styles.statItem}>
            <Text style={[styles.statValue, { color: t.accent }]}>{foilCount}</Text>
            <Text style={[styles.statLabel, { color: t.textSecondary }]}>Foils</Text>
          </View>
        </View>
      </View>

      {/* Theme */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Theme</Text>

        {/* Built-in pills */}
        <View style={[styles.themePills, styles.rowGap]}>
          {BUILT_IN_THEMES.map((th) => {
            const active = theme === th.name;
            return (
              <TouchableOpacity
                key={th.name}
                style={[
                  styles.themePill,
                  { backgroundColor: t.surface, borderColor: active ? t.accent : t.border },
                ]}
                onPress={() => setTheme(th.name)}
                accessibilityRole="button"
                accessibilityState={{ selected: active }}
              >
                <Text style={[styles.themePillText, { color: active ? t.accent : t.textSecondary }]}>
                  {th.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Custom slots */}
        <View style={styles.themePills}>
          {([0, 1, 2] as const).map((index) => {
            const customName = `custom-${index}` as ThemeName;
            const custom = customThemes[index];
            const active = theme === customName;

            if (custom) {
              return (
                <View
                  key={customName}
                  style={[
                    styles.themePill,
                    { backgroundColor: t.surface, borderColor: active ? t.accent : t.border },
                  ]}
                >
                  <TouchableOpacity
                    style={{ flex: 1 }}
                    onPress={() => setTheme(customName)}
                    accessibilityRole="button"
                    accessibilityState={{ selected: active }}
                    accessibilityLabel={custom.label}
                  >
                    <Text style={[styles.themePillText, { color: active ? t.accent : t.textSecondary }]} numberOfLines={1}>
                      {custom.label}
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => router.push(`/theme-editor?slot=${index}&mode=edit`)}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    accessibilityRole="button"
                    accessibilityLabel={`Edit ${custom.label}`}
                  >
                    <Text style={[styles.editIcon, { color: t.textSecondary }]}>✏️</Text>
                  </TouchableOpacity>
                </View>
              );
            }

            return (
              <TouchableOpacity
                key={customName}
                style={[
                  styles.themePill,
                  styles.themePillEmpty,
                  { borderColor: t.border },
                ]}
                onPress={() => router.push(`/theme-editor?slot=${index}&mode=new`)}
                accessibilityRole="button"
                accessibilityLabel={`Create custom theme ${index + 1}`}
              >
                <Text style={[styles.themePillText, { color: t.textSecondary }]}>+</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Feedback */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Feedback</Text>
        <View style={[styles.feedbackRow, { backgroundColor: t.surface }]}>
          <Text selectable style={[styles.feedbackText, { color: t.text }]}>
            Send feedback to{' '}
            <Text
              selectable
              style={{ color: t.accent, fontWeight: '600' }}
            >
              {FEEDBACK_EMAIL}
            </Text>
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, padding: 24, paddingTop: 60 },
  closeBtn: { position: 'absolute', top: 16, right: 16, padding: 8 },
  closeBtnText: { fontSize: 18 },
  screenTitle: { fontSize: 22, fontWeight: '700', marginBottom: 32 },

  section: { marginBottom: 28 },
  sectionLabel: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 10,
  },

  statsCard: {
    borderRadius: 12,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  statItem: { alignItems: 'center' },
  statValue: { fontSize: 22, fontWeight: '700' },
  statLabel: { fontSize: 12, marginTop: 4 },
  statDivider: { width: 1, height: 32 },

  themePills: { flexDirection: 'row', gap: 8 },
  rowGap: { marginBottom: 8 },
  themePill: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 4,
  },
  themePillEmpty: { borderStyle: 'dashed' },
  themePillText: { fontWeight: '600', fontSize: 13 },
  editIcon: { fontSize: 12 },

  feedbackRow: {
    borderRadius: 10,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  feedbackText: { fontSize: 14 },
  feedbackArrow: { fontSize: 18 },
});
