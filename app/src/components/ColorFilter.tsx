import { Modal, View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useState } from 'react';
import { useTheme } from '../theme/useTheme';
import { spacing, radius, font, MIN_TOUCH, HIT_SLOP_8 } from '../theme/themes';

export type ColorFilter = 'W' | 'U' | 'B' | 'R' | 'G' | 'C' | 'all';

const COLORS: { key: ColorFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'W', label: 'W' }, { key: 'U', label: 'U' }, { key: 'B', label: 'B' },
  { key: 'R', label: 'R' }, { key: 'G', label: 'G' }, { key: 'C', label: 'C' },
];

type Props = { active: ColorFilter; onChange: (c: ColorFilter) => void };

export function ColorFilter({ active, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const theme = useTheme();
  const isFiltered = active !== 'all';

  return (
    <>
      <TouchableOpacity
        style={[styles.btn, { backgroundColor: isFiltered ? theme.accent : theme.surfaceAlt }]}
        onPress={() => setOpen(true)}
        hitSlop={HIT_SLOP_8}
        accessibilityRole="button"
        accessibilityLabel={isFiltered ? `Filter color ${active}` : 'Open color filter'}
      >
        <Text style={[styles.btnText, { color: isFiltered ? theme.text : theme.textSecondary }]}>
          {isFiltered ? `Filter: ${active}` : 'Filters'}
        </Text>
      </TouchableOpacity>

      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={() => setOpen(false)}>
          <View style={[styles.sheet, { backgroundColor: theme.surface }]}>
            <Text style={[styles.title, { color: theme.text }]}>Filter by Color</Text>
            <View style={styles.grid}>
              {COLORS.map(({ key, label }) => {
                const selected = active === key;
                return (
                  <TouchableOpacity
                    key={key}
                    style={[styles.chip, { backgroundColor: selected ? theme.accent : theme.surfaceAlt }]}
                    onPress={() => { onChange(key); setOpen(false); }}
                    accessibilityRole="button"
                    accessibilityState={{ selected }}
                    accessibilityLabel={label}
                  >
                    <Text style={[styles.chipText, { color: selected ? theme.text : theme.textSecondary }]}>
                      {label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  btn: {
    alignSelf: 'flex-start',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.md,
    minHeight: MIN_TOUCH,
    justifyContent: 'center',
  },
  btnText: { fontSize: 13, fontWeight: '600' },
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  sheet: { borderTopLeftRadius: radius.xl, borderTopRightRadius: radius.xl, padding: spacing.xl, paddingBottom: 40 },
  title: { fontSize: font.subhead, fontWeight: '700', marginBottom: spacing.lg },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm + 2, justifyContent: 'center' },
  chip: {
    minWidth: MIN_TOUCH, minHeight: MIN_TOUCH,
    paddingHorizontal: spacing.lg, paddingVertical: spacing.sm + 2,
    borderRadius: radius.lg,
    alignItems: 'center', justifyContent: 'center',
  },
  chipText: { fontSize: font.body, fontWeight: '600' },
});
