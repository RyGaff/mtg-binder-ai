import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../theme/useTheme';
import type { SortMode } from '../utils/deckSections';

type Props = {
  mode: SortMode;
  dir: 'asc' | 'desc';
  onSelectMode: (mode: SortMode) => void;
  onToggleDir: () => void;
};

const MODES: { key: SortMode; label: string }[] = [
  { key: 'type', label: 'Type' },
  { key: 'name', label: 'Name' },
  { key: 'mana', label: 'Mana' },
  { key: 'color', label: 'Color' },
  { key: 'price', label: 'Price' },
];

function DeckSortPanelImpl({ mode, dir, onSelectMode, onToggleDir }: Props) {
  const t = useTheme();
  return (
    <View style={[s.wrap, { borderBottomColor: t.border, backgroundColor: t.surface }]}>
      <View style={s.row}>
        <Text style={[s.label, { color: t.textSecondary }]}>Sort by</Text>
        <View style={s.pills}>
          {MODES.map((m) => {
            const active = mode === m.key;
            return (
              <Pressable
                key={m.key}
                onPress={() => onSelectMode(m.key)}
                hitSlop={6}
                accessibilityRole="button"
                accessibilityState={{ selected: active }}
                style={[
                  s.pill,
                  {
                    backgroundColor: active ? t.accent + '33' : t.surfaceAlt,
                    borderColor: active ? t.accent : t.border,
                  },
                ]}
              >
                <Text style={[s.pillText, { color: active ? t.accent : t.text }]}>{m.label}</Text>
              </Pressable>
            );
          })}
        </View>
      </View>
      <View style={s.row}>
        <Text style={[s.label, { color: t.textSecondary }]}>Direction</Text>
        <View style={s.pills}>
          <Pressable
            onPress={() => dir === 'desc' && onToggleDir()}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel="Sort ascending"
            accessibilityState={{ selected: dir === 'asc' }}
            style={[s.pill, { backgroundColor: dir === 'asc' ? t.accent + '33' : t.surfaceAlt, borderColor: dir === 'asc' ? t.accent : t.border }]}
          >
            <Text style={[s.pillText, { color: dir === 'asc' ? t.accent : t.text }]}>↑ Asc</Text>
          </Pressable>
          <Pressable
            onPress={() => dir === 'asc' && onToggleDir()}
            hitSlop={6}
            accessibilityRole="button"
            accessibilityLabel="Sort descending"
            accessibilityState={{ selected: dir === 'desc' }}
            style={[s.pill, { backgroundColor: dir === 'desc' ? t.accent + '33' : t.surfaceAlt, borderColor: dir === 'desc' ? t.accent : t.border }]}
          >
            <Text style={[s.pillText, { color: dir === 'desc' ? t.accent : t.text }]}>↓ Desc</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

export const DeckSortPanel = memo(DeckSortPanelImpl);

const s = StyleSheet.create({
  wrap: { paddingVertical: 10, paddingHorizontal: 14, borderBottomWidth: 1, gap: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, flexWrap: 'wrap' },
  label: { fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6, minWidth: 70 },
  pills: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  pill: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999, borderWidth: StyleSheet.hairlineWidth },
  pillText: { fontSize: 12, fontWeight: '600' },
});
