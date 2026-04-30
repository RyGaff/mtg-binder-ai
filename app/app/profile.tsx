import { useCallback, useEffect, useState } from 'react';
import { ScrollView, View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { Platform } from 'react-native';
import { Image } from 'expo-image';
import { useStore } from '../src/store/useStore';
import { useTheme } from '../src/theme/useTheme';
import { getCollectionTotalValue, getFoilCount, getTotalCardCount } from '../src/db/collection';
import { clearSessionCardCache, getSessionCacheSize } from '../src/api/cards';
import { getDb } from '../src/db/db';
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

// Debug-stat row shape. `bytes` drives the sort + heaviest-first display;
// `count` is shown alongside when meaningful (rows / entries). `note` is a
// short qualifier when bytes are estimated rather than measured.
type DebugStat = { label: string; bytes?: number; count?: number; note?: string };

function fmtBytes(n: number | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// Hermes exposes a (semi-stable) instrumented-stats object. Shape varies by
// runtime version; we only read fields we actually use and cast through any.
type HermesStats = { js_totalAllocatedBytes?: number; js_heapSize?: number };

function gatherDebugStats(qc: QueryClient): DebugStat[] {
  const out: DebugStat[] = [];

  // ---- JS heap (Hermes) ----
  const hi = (globalThis as unknown as { HermesInternal?: { getInstrumentedStats?: () => HermesStats } }).HermesInternal;
  const hStats = hi?.getInstrumentedStats?.();
  if (hStats?.js_heapSize != null) {
    out.push({ label: 'JS heap (Hermes)', bytes: hStats.js_heapSize, note: 'live' });
  } else if (hStats?.js_totalAllocatedBytes != null) {
    out.push({ label: 'JS heap (Hermes)', bytes: hStats.js_totalAllocatedBytes, note: 'allocated' });
  }

  // ---- Session card LRU ----
  // Each cached card is ~2 KB of JSON. 50-cap → ~100 KB max.
  const lruCount = getSessionCacheSize();
  out.push({ label: 'Session card LRU', count: lruCount, bytes: lruCount * 2048, note: 'est' });

  // ---- TanStack Query cache ----
  // Estimate by JSON-encoding each query's data and summing byte length.
  // Cheap enough to run on demand; we don't poll continuously.
  let qBytes = 0;
  const queries = qc.getQueryCache().getAll();
  for (const q of queries) {
    if (q.state.data === undefined) continue;
    try { qBytes += JSON.stringify(q.state.data).length; } catch { /* circular: skip */ }
  }
  out.push({ label: 'TanStack Query cache', count: queries.length, bytes: qBytes, note: 'est' });

  // ---- SQLite (file size + per-table row counts) ----
  try {
    const db = getDb();
    const pc = db.getFirstSync<{ page_count: number }>('PRAGMA page_count');
    const ps = db.getFirstSync<{ page_size: number }>('PRAGMA page_size');
    const dbBytes = (pc?.page_count ?? 0) * (ps?.page_size ?? 0);
    out.push({ label: 'SQLite db file', bytes: dbBytes, note: 'on disk' });
    for (const tbl of ['cards', 'printings', 'deck_history', 'collection_entries']) {
      const r = db.getFirstSync<{ n: number }>(`SELECT COUNT(*) AS n FROM ${tbl}`);
      out.push({ label: `  ↳ ${tbl}`, count: r?.n ?? 0 });
    }
  } catch (e) {
    out.push({ label: 'SQLite db file', note: `error: ${(e as Error).message}` });
  }

  // ---- Decoded image bitmaps (expo-image / SDWebImage) ----
  // No public API exposes the live decoded-bitmap pool size, so we report
  // the configured caps instead. These are upper bounds — actual usage is
  // somewhere between 0 and the cap depending on what's been rendered. iOS
  // values come from Image.configureCache() in app/_layout.tsx; Android uses
  // Glide defaults which we don't bound.
  if (Platform.OS === 'ios') {
    out.push({ label: 'Image bitmap pool (iOS)', bytes: 100 * 1024 * 1024, note: 'cap' });
    out.push({ label: '  ↳ entries', count: 200, note: 'cap' });
    out.push({ label: 'Image disk cache (iOS)', bytes: 250 * 1024 * 1024, note: 'cap' });
  } else {
    out.push({ label: 'Image bitmap pool', note: 'unbounded (Glide default)' });
  }

  // ---- In-memory store buckets ----
  const recentScansLen = useStore.getState().recentScans.length;
  const trailLen = useStore.getState().cardTrail.length;
  out.push({ label: 'Recent scans', count: recentScansLen });
  out.push({ label: 'Card trail', count: trailLen });

  // Heaviest first; rows without bytes (count-only / sub-rows) sink to the end
  // so the user's eye lands on byte-pressure first.
  return out.sort((a, b) => (b.bytes ?? -1) - (a.bytes ?? -1));
}

export default function ProfileScreen() {
  const router = useRouter();
  const qc = useQueryClient();
  const [cacheStatus, setCacheStatus] = useState<'idle' | 'clearing' | 'cleared'>('idle');
  const [debugStats, setDebugStats] = useState<DebugStat[]>([]);
  const refreshDebugStats = useCallback(() => { setDebugStats(gatherDebugStats(qc)); }, [qc]);
  // Initial poll on mount + re-poll whenever the cache-clear flow finishes
  // (so the user sees their wipe reflected without manually refreshing).
  useEffect(() => { refreshDebugStats(); }, [refreshDebugStats, cacheStatus]);
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

  // Clear all transient caches: expo-image (memory + disk), the in-memory
  // session card LRU, and TanStack Query results. SQLite tables (cards,
  // decks, collection) are user data and stay untouched.
  const clearCache = async () => {
    if (cacheStatus === 'clearing') return;
    setCacheStatus('clearing');
    try {
      clearSessionCardCache();
      qc.clear();
      await Promise.allSettled([
        Image.clearMemoryCache(),
        Image.clearDiskCache(),
      ]);
      setCacheStatus('cleared');
      setTimeout(() => setCacheStatus('idle'), 1500);
    } catch (e) {
      console.warn('clearCache failed', e);
      setCacheStatus('idle');
    }
  };
  const pillStyle = (active: boolean) =>
    [styles.themePill, { backgroundColor: t.surface, borderColor: active ? t.accent : t.border }];
  const pillTextStyle = (active: boolean) =>
    [styles.themePillText, { color: active ? t.accent : t.textSecondary }];

  return (
    <View style={[styles.screen, { backgroundColor: t.bg }]}>
      {/* Close button stays anchored to the screen (outside the scroll view)
          so it's always reachable regardless of how far the user has scrolled. */}
      <TouchableOpacity
        style={styles.closeBtn}
        onPress={closeScreen}
        hitSlop={HIT_SLOP}
        accessibilityLabel="Close"
        accessibilityRole="button"
      >
        <Icon name="close" size={20} color={t.textSecondary} />
      </TouchableOpacity>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
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

      {/* Storage / cache */}
      <View style={styles.section}>
        <Text style={[styles.sectionLabel, { color: t.textSecondary }]}>Storage</Text>
        <TouchableOpacity
          onPress={() => { void clearCache(); }}
          disabled={cacheStatus === 'clearing'}
          style={[styles.themePill, { backgroundColor: t.surface, borderColor: t.border }]}
          accessibilityRole="button"
          accessibilityLabel="Clear cache"
        >
          <Text style={[styles.themePillText, { color: cacheStatus === 'cleared' ? t.accent : t.text }]}>
            {cacheStatus === 'clearing' ? 'Clearing…' : cacheStatus === 'cleared' ? 'Cleared' : 'Clear cache'}
          </Text>
        </TouchableOpacity>
        <Text style={[styles.cacheHint, { color: t.textSecondary }]}>
          Drops cached card images and search results. Your collection, decks, and themes are not affected.
        </Text>

        {/* Debug — what's currently sitting in RAM / on disk. Sorted by
            byte-pressure so the heaviest bucket reads at the top. Refresh
            polls a fresh snapshot; values are cheap to gather (cap ~50ms). */}
        <View style={[styles.debugHeader, { borderTopColor: t.border }]}>
          <Text style={[styles.sectionLabel, { color: t.textSecondary, marginBottom: 0 }]}>Debug · memory</Text>
          <TouchableOpacity onPress={refreshDebugStats} hitSlop={HIT_SLOP} accessibilityRole="button" accessibilityLabel="Refresh memory stats">
            <Text style={[styles.debugRefresh, { color: t.accent }]}>Refresh</Text>
          </TouchableOpacity>
        </View>
        <View style={[styles.debugCard, { backgroundColor: t.surface, borderColor: t.border }]}>
          {debugStats.length === 0 ? (
            <Text style={[styles.cacheHint, { color: t.textSecondary }]}>No data yet.</Text>
          ) : debugStats.map((stat) => (
            <View key={stat.label} style={styles.debugRow}>
              <Text style={[styles.debugLabel, { color: t.text }]} numberOfLines={1}>{stat.label}</Text>
              <View style={styles.debugRight}>
                {stat.count != null ? (
                  <Text style={[styles.debugCount, { color: t.textSecondary }]}>{stat.count}</Text>
                ) : null}
                {stat.bytes != null ? (
                  <Text style={[styles.debugBytes, { color: t.text }]}>{fmtBytes(stat.bytes)}</Text>
                ) : null}
                {stat.note ? (
                  <Text style={[styles.debugNote, { color: t.textSecondary }]}>{stat.note}</Text>
                ) : null}
              </View>
            </View>
          ))}
        </View>
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
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  // Outer screen now hosts the ScrollView; padding moved to scrollContent so
  // the scroll surface fills the viewport and the bottom inset clears
  // anything below the last section.
  screen: { flex: 1 },
  scroll: { flex: 1 },
  scrollContent: { padding: 24, paddingTop: 60, paddingBottom: 48 },
  closeBtn: { position: 'absolute', top: 16, right: 16, padding: 8, minWidth: 44, minHeight: 44, alignItems: 'flex-end', justifyContent: 'center', zIndex: 1 },
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
  cacheHint: { fontSize: 12, marginTop: 8, lineHeight: 16 },
  debugHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 16, paddingTop: 12, borderTopWidth: StyleSheet.hairlineWidth },
  debugRefresh: { fontSize: 12, fontWeight: '700' },
  debugCard: { borderRadius: 10, borderWidth: 1, paddingVertical: 6, paddingHorizontal: 10, marginTop: 8 },
  debugRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 6 },
  debugLabel: { flex: 1, fontSize: 12, fontWeight: '600' },
  debugRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  debugCount: { fontSize: 11, fontVariant: ['tabular-nums'] },
  debugBytes: { fontSize: 12, fontWeight: '700', fontVariant: ['tabular-nums'], minWidth: 64, textAlign: 'right' },
  debugNote: { fontSize: 10, fontStyle: 'italic', minWidth: 44, textAlign: 'right' },
});
