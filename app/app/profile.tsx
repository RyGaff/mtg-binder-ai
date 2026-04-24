import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useStore } from '../src/store/useStore';
import { useTheme } from '../src/theme/useTheme';
import { getCollectionTotalValue, getFoilCount, getTotalCardCount } from '../src/db/collection';
import type { ThemeName } from '../src/theme/themes';
import type { SearchViewMode, SearchGridCols } from '../src/store/useStore';
import { Icon } from '../src/components/icons/Icon';

const BUILT_IN_THEMES: { name: ThemeName; label: string }[] = [
  { name: 'dark', label: 'Dark' },
  { name: 'light', label: 'Light' },
  { name: 'amoled', label: 'AMOLED' },
];

const VIEW_MODES: { mode: SearchViewMode; label: string }[] = [
  { mode: 'list', label: 'List' },
  { mode: 'grid', label: 'Grid' },
];

const GRID_COL_OPTIONS: SearchGridCols[] = [1, 2, 3, 4, 5];
const FEEDBACK_EMAIL = 'lotusfieldmtg@gmail.com';
const HIT_SLOP = { top: 8, bottom: 8, left: 8, right: 8 };

export default function ProfileScreen() {
  const router = useRouter();
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const customThemes = useStore((s) => s.customThemes);
  const searchViewMode = useStore((s) => s.searchViewMode);
  const setSearchViewMode = useStore((s) => s.setSearchViewMode);
  const searchGridCols = useStore((s) => s.searchGridCols);
  const setSearchGridCols = useStore((s) => s.setSearchGridCols);
  const t = useTheme();

  const { data: totalValue = 0 } = useQuery({ queryKey: ['collection-value'], queryFn: getCollectionTotalValue });
  const { data: totalCards = 0 } = useQuery({ queryKey: ['total-cards'], queryFn: getTotalCardCount });
  const { data: foilCount = 0 } = useQuery({ queryKey: ['foil-count'], queryFn: getFoilCount });

  const closeScreen = () => (router.canGoBack() ? router.back() : router.replace('/'));
  const pillStyle = (active: boolean) =>
    [styles.themePill, { backgroundColor: t.surface, borderColor: active ? t.accent : t.border }];
  const pillTextStyle = (active: boolean) =>
    [styles.themePillText, { color: active ? t.accent : t.textSecondary }];

  return (
    <View style={[styles.screen, { backgroundColor: t.bg }]}>
      <TouchableOpacity
        style={styles.closeBtn}
        onPress={closeScreen}
        hitSlop={HIT_SLOP}
        accessibilityLabel="Close"
        accessibilityRole="button"
      >
        <Icon name="close" size={20} color={t.textSecondary} />
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

        <View style={[styles.themePills, styles.rowGap]}>
          {BUILT_IN_THEMES.map((th) => {
            const active = theme === th.name;
            return (
              <TouchableOpacity
                key={th.name}
                style={pillStyle(active)}
                onPress={() => setTheme(th.name)}
                accessibilityRole="button"
                accessibilityState={{ selected: active }}
              >
                <Text style={pillTextStyle(active)}>{th.label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>

        <View style={styles.themePills}>
          {([0, 1, 2] as const).map((index) => {
            const customName = `custom-${index}` as ThemeName;
            const custom = customThemes[index];
            const active = theme === customName;

            if (custom) {
              return (
                <View key={customName} style={pillStyle(active)}>
                  <TouchableOpacity
                    style={{ flex: 1 }}
                    onPress={() => setTheme(customName)}
                    accessibilityRole="button"
                    accessibilityState={{ selected: active }}
                    accessibilityLabel={custom.label}
                  >
                    <Text style={pillTextStyle(active)} numberOfLines={1}>{custom.label}</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => router.push(`/theme-editor?slot=${index}&mode=edit`)}
                    hitSlop={HIT_SLOP}
                    accessibilityRole="button"
                    accessibilityLabel={`Edit ${custom.label}`}
                  >
                    <Icon name="pencil" size={14} color={t.textSecondary} />
                  </TouchableOpacity>
                </View>
              );
            }

            return (
              <TouchableOpacity
                key={customName}
                style={[styles.themePill, styles.themePillEmpty, { borderColor: t.border }]}
                onPress={() => router.push(`/theme-editor?slot=${index}&mode=new`)}
                accessibilityRole="button"
                accessibilityLabel={`Create custom theme ${index + 1}`}
              >
                <Icon name="plus" size={16} color={t.textSecondary} />
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Search view */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Search view</Text>
        <View style={[styles.themePills, styles.rowGap]}>
          {VIEW_MODES.map((v) => {
            const active = searchViewMode === v.mode;
            return (
              <TouchableOpacity
                key={v.mode}
                style={pillStyle(active)}
                onPress={() => setSearchViewMode(v.mode)}
                accessibilityRole="button"
                accessibilityState={{ selected: active }}
              >
                <Text style={pillTextStyle(active)}>{v.label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
        {searchViewMode === 'grid' && (
          <View style={styles.themePills}>
            {GRID_COL_OPTIONS.map((n) => {
              const active = searchGridCols === n;
              return (
                <TouchableOpacity
                  key={n}
                  style={pillStyle(active)}
                  onPress={() => setSearchGridCols(n)}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  accessibilityLabel={`${n} cards per row`}
                >
                  <Text style={pillTextStyle(active)}>{n}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        )}
      </View>

      {/* Feedback */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Feedback (TestFlight)</Text>
        <View style={[styles.feedbackRow, { backgroundColor: t.surface }]}>
          <Text style={[styles.feedbackText, { color: t.text }]}>
            <Text style={{ fontWeight: '600' }}>Screenshot:</Text> take a screenshot anywhere in the app,
            then tap <Text style={{ fontWeight: '600' }}>Share Beta Feedback</Text> in the preview. Annotate
            and send — it goes straight to the dev.
            {'\n\n'}
            <Text style={{ fontWeight: '600' }}>Crashes:</Text> auto-sent after you reopen the app. Tap
            <Text style={{ fontWeight: '600' }}> Share</Text> when prompted to include details.
            {'\n\n'}
            <Text style={{ fontWeight: '600' }}>Other issues:</Text> open the TestFlight app → MTG Binder AI →
            <Text style={{ fontWeight: '600' }}> Send Beta Feedback</Text>.
            {'\n\n'}
            Prefer email? <Text selectable style={{ color: t.accent, fontWeight: '600' }}>{FEEDBACK_EMAIL}</Text>
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, padding: 24, paddingTop: 60 },
  closeBtn: { position: 'absolute', top: 16, right: 16, padding: 8, minWidth: 44, minHeight: 44, alignItems: 'flex-end', justifyContent: 'center' },
  screenTitle: { fontSize: 22, fontWeight: '700', marginBottom: 32 },
  section: { marginBottom: 28 },
  sectionLabel: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 },
  statsCard: { borderRadius: 12, padding: 20, flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' },
  statItem: { alignItems: 'center' },
  statValue: { fontSize: 22, fontWeight: '700' },
  statLabel: { fontSize: 12, marginTop: 4 },
  statDivider: { width: 1, height: 32 },
  themePills: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  rowGap: { marginBottom: 8 },
  themePill: {
    flexGrow: 1,
    flexBasis: 80,
    minHeight: 44,
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
  feedbackRow: { borderRadius: 10, padding: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  feedbackText: { fontSize: 14 },
});
