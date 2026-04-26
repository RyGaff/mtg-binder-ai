import { memo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { manaGlyph, manaTint, parseManaCost, MANA_FACE_SEP } from '../utils/mana';

type Props = { cost: string | null | undefined; size?: number };

function ManaCostImpl({ cost, size = 12 }: Props) {
  const tokens = parseManaCost(cost);
  if (tokens.length === 0) return null;
  return (
    <View style={s.row}>
      {tokens.map((t, i) => {
        if (t === MANA_FACE_SEP) {
          return <Text key={i} style={[s.sep, { fontSize: size, lineHeight: size + 2 }]}>{'//'}</Text>;
        }
        const g = manaGlyph(t);
        if (g == null) return <Text key={i} style={[s.fallback, { fontSize: size }]}>{`{${t}}`}</Text>;
        return (
          <Text key={i} style={{ fontFamily: 'Mana', color: manaTint(t), fontSize: size, lineHeight: size + 2 }}>
            {g}
          </Text>
        );
      })}
    </View>
  );
}

export const ManaCost = memo(ManaCostImpl);

const s = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 1 },
  fallback: { fontFamily: 'monospace', color: '#a4abbb' },
  sep: { color: '#a4abbb', marginHorizontal: 3 },
});
