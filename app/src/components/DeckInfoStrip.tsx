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
  expanded: boolean;
  onToggleStats: () => void;
};

function DeckInfoStripImpl({ format, colorIdentity, mainCount, sideCount, totalPrice, expanded, onToggleStats }: Props) {
  const t = useTheme();
  return (
    <View style={[s.wrap, { borderBottomColor: t.border }]}>
      <View style={[s.pill, { backgroundColor: t.surfaceAlt }]}>
        <Text style={[s.pillText, { color: t.textSecondary }]}>{format}</Text>
      </View>
      <View style={s.glyphs}>
        {colorIdentity.map((c) => (
          <Text key={c} style={{ fontFamily: 'Mana', color: manaTint(c), fontSize: 14, lineHeight: 16 }}>
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
      <Pressable onPress={onToggleStats} style={[s.toggle, expanded ? { backgroundColor: t.accent + '33' } : { backgroundColor: t.surfaceAlt }]}>
        <Text style={[s.toggleText, { color: expanded ? t.accent : t.textSecondary }]}>
          Stats {expanded ? '▴' : '▾'}
        </Text>
      </Pressable>
    </View>
  );
}

export const DeckInfoStrip = memo(DeckInfoStripImpl);

const s = StyleSheet.create({
  wrap: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 8, paddingHorizontal: 14, paddingVertical: 10, borderBottomWidth: 1 },
  pill: { paddingHorizontal: 9, paddingVertical: 3, borderRadius: 999 },
  pillText: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
  glyphs: { flexDirection: 'row', alignItems: 'center', gap: 2 },
  count: { fontSize: 11 },
  price: { fontSize: 11, fontWeight: '700' },
  toggle: { marginLeft: 'auto', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  toggleText: { fontSize: 10, fontWeight: '600' },
});
