import { Modal, View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useState } from 'react';
import { useTheme } from '../theme/useTheme';

export type ColorFilter = 'W' | 'U' | 'B' | 'R' | 'G' | 'C' | 'all';

const COLORS: { key: ColorFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'W', label: 'W' },
  { key: 'U', label: 'U' },
  { key: 'B', label: 'B' },
  { key: 'R', label: 'R' },
  { key: 'G', label: 'G' },
  { key: 'C', label: 'C' },
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
              {COLORS.map(({ key, label }) => (
                <TouchableOpacity
                  key={key}
                  style={[styles.chip, { backgroundColor: active === key ? theme.accent : theme.surfaceAlt }]}
                  onPress={() => { onChange(key); setOpen(false); }}
                >
                  <Text style={[styles.chipText, { color: active === key ? theme.text : theme.textSecondary }]}>
                    {label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  btn: { alignSelf: 'flex-start', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 8 },
  btnText: { fontSize: 13, fontWeight: '600' },
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  sheet: { borderTopLeftRadius: 16, borderTopRightRadius: 16, padding: 24, paddingBottom: 40 },
  title: { fontSize: 15, fontWeight: '700', marginBottom: 16 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  chip: { paddingHorizontal: 20, paddingVertical: 8, borderRadius: 12 },
  chipText: { fontSize: 14, fontWeight: '600' },
});
