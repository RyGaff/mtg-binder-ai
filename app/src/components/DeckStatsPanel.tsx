import { memo, useMemo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { DeckCard } from '../db/decks';
import { useTheme } from '../theme/useTheme';
import { avgCmc, boardPrice, colorPipCounts, manaCurve, typeCounts } from '../utils/deckStats';
import { CARD_TYPE_ORDER } from '../utils/cardHelpers';
import { MANA_TINT, manaGlyph } from '../utils/mana';

type Props = { main: DeckCard[]; commander: DeckCard[]; side: DeckCard[]; format: string };

function DeckStatsPanelImpl({ main, commander, side, format }: Props) {
  const t = useTheme();
  // Curve / colors / types / avg always combine main + commander — the commander is a
  // real card that contributes to those distributions regardless of format. Price is
  // the only stat that distinguishes whether the commander counts toward "main".
  const mainCommander = useMemo(() => [...main, ...commander], [main, commander]);
  const curve = useMemo(() => manaCurve(mainCommander), [mainCommander]);
  const pips = useMemo(() => colorPipCounts(mainCommander), [mainCommander]);
  const types = useMemo(() => typeCounts(mainCommander), [mainCommander]);
  const avg = useMemo(() => avgCmc(mainCommander), [mainCommander]);
  // Main price rolls in the commander only for Commander-format decks. For other
  // formats (Standard, Modern, etc.) the commander board is rare/edge and shouldn't
  // inflate the main-deck price.
  const isCommanderFmt = format.trim().toLowerCase() === 'commander';
  const mainPrice = useMemo(
    () => boardPrice(main) + (isCommanderFmt ? boardPrice(commander) : 0),
    [main, commander, isCommanderFmt],
  );
  const sidePrice = useMemo(() => boardPrice(side), [side]);

  const max = Math.max(1, ...Object.values(curve));
  const buckets: (keyof typeof curve)[] = [0, 1, 2, 3, 4, 5, '6+'];
  const totalPip = (['W','U','B','R','G','C'] as const).reduce((s, k) => s + pips[k], 0) || 1;

  return (
    <View style={[s.wrap, { backgroundColor: t.surface, borderBottomColor: t.border }]}>
      {/* Mana-curve header doubles as the Avg CMC readout — keeps the average visually
          tied to the curve it summarizes instead of stranded down in the numeric row. */}
      <View style={s.curveHeader}>
        <Text style={[s.h4, { color: t.textSecondary }]}>Mana curve</Text>
        <Text style={[s.curveAvg, { color: t.textSecondary }]}>
          Avg CMC <Text style={{ color: t.text, fontWeight: '600' }}>{avg.toFixed(2)}</Text>
        </Text>
      </View>
      <View style={s.curve}>
        {buckets.map((b) => (
          <View key={String(b)} style={s.col}>
            {/* Count sits above the bar so the visual hierarchy is value → bar → bucket. */}
            <Text style={[s.count, { color: t.text }]}>{curve[b]}</Text>
            <View style={[s.bar, { height: 8 + (curve[b] / max) * 36, backgroundColor: t.accent }]} />
            <Text style={[s.lbl, { color: t.textSecondary }]}>{String(b)}</Text>
          </View>
        ))}
      </View>

      <Text style={[s.h4, { color: t.textSecondary }]}>Color identity</Text>
      <View style={s.pipBar}>
        {(['W','U','B','R','G','C'] as const).map((k) => (
          pips[k] > 0 ? <View key={k} style={{ backgroundColor: MANA_TINT[k], flex: pips[k] / totalPip }} /> : null
        ))}
      </View>
      {/* Percent breakdown — one chip per color present, sized to mirror its bar share
          so labels align under their segment. Rounded for legibility, not exact. The
          Mana-font glyph is rendered alongside the percent so the cue is visual, not
          letter-coded. */}
      <View style={s.pipPctRow}>
        {(['W','U','B','R','G','C'] as const).map((k) => (
          pips[k] > 0 ? (
            <View key={k} style={[s.pipPctCell, { flex: pips[k] / totalPip }]}>
              <Text style={[s.pipGlyph, { color: MANA_TINT[k] }]} numberOfLines={1}>
                {manaGlyph(k) ?? k}
              </Text>
              <Text style={[s.pipPctText, { color: t.textSecondary }]} numberOfLines={1}>
                {Math.round((pips[k] / totalPip) * 100)}%
              </Text>
            </View>
          ) : null
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
        <Text style={[s.numItem, { color: t.textSecondary }]}>Lands <Text style={[s.numVal, { color: t.text }]}>{types.Land}</Text></Text>
        {mainPrice > 0 ? (
          <Text style={[s.numItem, { color: t.textSecondary }]}>Main <Text style={[s.numVal, { color: t.text }]}>${mainPrice.toFixed(2)}</Text></Text>
        ) : null}
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
  // Header row for the curve so "Avg X.YZ" can sit on the right inline with the title.
  curveHeader: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  curveAvg: { fontSize: 11 },
  // Curve grows taller to fit count-above-bar layout (was 60). Cols still anchor to
  // the bottom so bars line up along a common baseline regardless of count text.
  curve: { flexDirection: 'row', alignItems: 'flex-end', gap: 4, height: 76 },
  col: { flex: 1, alignItems: 'center', gap: 3 },
  bar: { width: '70%', borderRadius: 2 },
  lbl: { fontSize: 9 },
  // Per-column count sitting above the bar — slightly bolder/bigger than the bucket
  // label so the eye reads the magnitude before the mana value.
  count: { fontSize: 11, fontWeight: '700' },
  pipBar: { flexDirection: 'row', height: 8, borderRadius: 4, overflow: 'hidden' },
  // Percent labels row under the pip bar — each cell flexes the same share as its
  // segment above, so labels visually anchor to their colors. Tight gap, small text.
  pipPctRow: { flexDirection: 'row', marginTop: -2, paddingHorizontal: 2 },
  // Glyph + percent on one horizontal row inside each cell — symbol left, number right.
  pipPctCell: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 2, gap: 3 },
  // Mana font glyph for the color — colored via MANA_TINT for visual recognition.
  pipGlyph: { fontFamily: 'Mana', fontSize: 12, lineHeight: 14 },
  pipPctText: { fontSize: 10, fontWeight: '600' },
  typesGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  typeRow: { width: '50%', flexDirection: 'row', justifyContent: 'space-between', paddingRight: 12, paddingVertical: 1 },
  typeName: { fontSize: 11 },
  typeCt: { fontSize: 11, fontWeight: '700' },
  nums: { flexDirection: 'row', gap: 14, flexWrap: 'wrap', marginTop: 2 },
  numItem: { fontSize: 11 },
  numVal: { fontWeight: '700' },
});
