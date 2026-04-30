import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../theme/useTheme';
import { manaGlyph, manaTint } from '../utils/mana';

type Props = {
  format: string;
  colorIdentity: string[]; // ['W','U','B','G']
  mainCount: number;
  sideCount: number;
  totalPrice: number;
  statsExpanded: boolean;
  historyExpanded: boolean;
  /** Active sort mode label shown next to the Sort toggle (e.g. "Mana"). */
  sortLabel: string;
  /** Active sort direction — drives the arrow shown on the Sort toggle. */
  sortDir: 'asc' | 'desc';
  /** Whether the Sort panel is currently expanded. Mutually exclusive with
      Stats and History (parent enforces the exclusion). */
  sortExpanded: boolean;
  onToggleStats: () => void;
  onToggleHistory: () => void;
  onToggleSort: () => void;
};

function DeckInfoStripImpl({
  format, colorIdentity, mainCount, sideCount, totalPrice,
  statsExpanded, historyExpanded, sortLabel, sortDir, sortExpanded,
  onToggleStats, onToggleHistory, onToggleSort,
}: Props) {
  const t = useTheme();
  return (
    <View style={[s.wrap, { borderBottomColor: t.border }]}>
      <View style={[s.pill, { backgroundColor: t.surfaceAlt }]}>
        <Text style={[s.pillText, { color: t.textSecondary }]}>{format}</Text>
      </View>
      <View style={s.glyphs}>
        {colorIdentity.map((c) => (
          <Text key={c} style={{ fontFamily: 'Mana', color: manaTint(c), fontSize: 16, lineHeight: 18 }}>
            {manaGlyph(c) ?? ''}
          </Text>
        ))}
      </View>
      <Text style={[s.count, { color: t.textSecondary }]}>
        <Text style={{ color: t.text, fontWeight: '700' }}>{mainCount}</Text> main · <Text style={{ color: t.text, fontWeight: '700' }}>{sideCount}</Text> side
      </Text>
      {totalPrice > 0 ? (
        <Text style={[s.price, { color: t.text }]}>
          <Text style={{ color: t.textSecondary, fontWeight: '500' }}>$</Text>{totalPrice.toFixed(2)}
        </Text>
      ) : null}
      {/* Toggle group pinned to the right edge. All three behave the same
          way now — they expand/collapse an inline panel below the strip.
          The parent screen enforces mutual exclusion so only one panel
          renders at a time. The Sort toggle's label echoes the active mode
          + direction so the user can read current state without opening it. */}
      <View style={s.toggleGroup}>
        <Pressable onPress={onToggleSort} hitSlop={8} style={[s.toggle, sortExpanded ? { backgroundColor: t.accent + '33' } : { backgroundColor: t.surfaceAlt }]}>
          <Text style={[s.toggleText, { color: sortExpanded ? t.accent : t.textSecondary }]}>
            Sort: {sortLabel} {sortDir === 'asc' ? '↑' : '↓'} {sortExpanded ? '▴' : '▾'}
          </Text>
        </Pressable>
        <Pressable onPress={onToggleStats} hitSlop={8} style={[s.toggle, statsExpanded ? { backgroundColor: t.accent + '33' } : { backgroundColor: t.surfaceAlt }]}>
          <Text style={[s.toggleText, { color: statsExpanded ? t.accent : t.textSecondary }]}>
            Stats {statsExpanded ? '▴' : '▾'}
          </Text>
        </Pressable>
        <Pressable onPress={onToggleHistory} hitSlop={8} style={[s.toggle, historyExpanded ? { backgroundColor: t.accent + '33' } : { backgroundColor: t.surfaceAlt }]}>
          <Text style={[s.toggleText, { color: historyExpanded ? t.accent : t.textSecondary }]}>
            History {historyExpanded ? '▴' : '▾'}
          </Text>
        </Pressable>
      </View>
    </View>
  );
}

export const DeckInfoStrip = memo(DeckInfoStripImpl);

const s = StyleSheet.create({
  wrap: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 8, paddingHorizontal: 14, paddingVertical: 12, borderBottomWidth: 1 },
  pill: { paddingHorizontal: 9, paddingVertical: 3, borderRadius: 999 },
  pillText: { fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
  glyphs: { flexDirection: 'row', alignItems: 'center', gap: 2 },
  count: { fontSize: 13 },
  price: { fontSize: 13, fontWeight: '700' },
  toggleGroup: { marginLeft: 'auto', flexDirection: 'row', alignItems: 'center', gap: 6 },
  toggle: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999 },
  toggleText: { fontSize: 12, fontWeight: '600' },
});
