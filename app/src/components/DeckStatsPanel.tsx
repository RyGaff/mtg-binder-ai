import { memo, useMemo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { DeckCard } from '../db/decks';
import { useTheme } from '../theme/useTheme';
import { avgCmc, boardPrice, colorPipCounts, manaCurve, typeCounts } from '../utils/deckStats';
import { CARD_TYPE_ORDER } from '../utils/cardHelpers';
import { MANA_TINT } from '../utils/mana';

type Props = { mainCommander: DeckCard[]; side: DeckCard[] };

function DeckStatsPanelImpl({ mainCommander, side }: Props) {
  const t = useTheme();
  const curve = useMemo(() => manaCurve(mainCommander), [mainCommander]);
  const pips = useMemo(() => colorPipCounts(mainCommander), [mainCommander]);
  const types = useMemo(() => typeCounts(mainCommander), [mainCommander]);
  const avg = useMemo(() => avgCmc(mainCommander), [mainCommander]);
  const sidePrice = useMemo(() => boardPrice(side), [side]);

  const max = Math.max(1, ...Object.values(curve));
  const buckets: (keyof typeof curve)[] = [0, 1, 2, 3, 4, 5, '6+'];
  const totalPip = (['W','U','B','R','G','C'] as const).reduce((s, k) => s + pips[k], 0) || 1;

  return (
    <View style={[s.wrap, { backgroundColor: t.surface, borderBottomColor: t.border }]}>
      <Text style={[s.h4, { color: t.textSecondary }]}>Mana curve</Text>
      <View style={s.curve}>
        {buckets.map((b) => (
          <View key={String(b)} style={s.col}>
            <View style={[s.bar, { height: 8 + (curve[b] / max) * 36, backgroundColor: t.accent }]} />
            <Text style={[s.lbl, { color: t.textSecondary }]}>{String(b)}</Text>
            <Text style={[s.lbl, { color: t.textSecondary }]}>{curve[b]}</Text>
          </View>
        ))}
      </View>

      <Text style={[s.h4, { color: t.textSecondary }]}>Color identity</Text>
      <View style={s.pipBar}>
        {(['W','U','B','R','G','C'] as const).map((k) => (
          pips[k] > 0 ? <View key={k} style={{ backgroundColor: MANA_TINT[k], flex: pips[k] / totalPip }} /> : null
        ))}
      </View>

      <Text style={[s.h4, { color: t.textSecondary }]}>By type</Text>
      <View style={s.typesGrid}>
        {CARD_TYPE_ORDER.filter((k) => types[k] > 0).map((k) => (
          <View key={k} style={s.typeRow}>
            <Text style={[s.typeName, { color: t.textSecondary }]}>{k}</Text>
            <Text style={[s.typeCt, { color: t.text }]}>{types[k]}</Text>
          </View>
        ))}
      </View>

      <View style={s.nums}>
        <Text style={[s.numItem, { color: t.textSecondary }]}>Avg CMC <Text style={[s.numVal, { color: t.text }]}>{avg.toFixed(2)}</Text></Text>
        <Text style={[s.numItem, { color: t.textSecondary }]}>Lands <Text style={[s.numVal, { color: t.text }]}>{types.Land}</Text></Text>
        {sidePrice > 0 ? (
          <Text style={[s.numItem, { color: t.textSecondary }]}>Sideboard <Text style={[s.numVal, { color: t.text }]}>${sidePrice.toFixed(2)}</Text></Text>
        ) : null}
      </View>
    </View>
  );
}

export const DeckStatsPanel = memo(DeckStatsPanelImpl);

const s = StyleSheet.create({
  wrap: { paddingHorizontal: 14, paddingVertical: 12, gap: 10, borderBottomWidth: 1 },
  h4: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.8 },
  curve: { flexDirection: 'row', alignItems: 'flex-end', gap: 4, height: 60 },
  col: { flex: 1, alignItems: 'center', gap: 3 },
  bar: { width: '70%', borderRadius: 2 },
  lbl: { fontSize: 9 },
  pipBar: { flexDirection: 'row', height: 8, borderRadius: 4, overflow: 'hidden' },
  typesGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  typeRow: { width: '50%', flexDirection: 'row', justifyContent: 'space-between', paddingRight: 12, paddingVertical: 1 },
  typeName: { fontSize: 11 },
  typeCt: { fontSize: 11, fontWeight: '700' },
  nums: { flexDirection: 'row', gap: 14, flexWrap: 'wrap', marginTop: 2 },
  numItem: { fontSize: 11 },
  numVal: { fontWeight: '700' },
});
