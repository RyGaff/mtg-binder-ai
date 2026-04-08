import { Modal, View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useState } from 'react';

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

  const isFiltered = active !== 'all';

  return (
    <>
      <TouchableOpacity
        style={[styles.btn, isFiltered && styles.btnActive]}
        onPress={() => setOpen(true)}
      >
        <Text style={[styles.btnText, isFiltered && styles.btnTextActive]}>
          {isFiltered ? `Filter: ${active}` : 'Filters'}
        </Text>
      </TouchableOpacity>

      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={() => setOpen(false)}>
          <View style={styles.sheet}>
            <Text style={styles.title}>Filter by Color</Text>
            <View style={styles.grid}>
              {COLORS.map(({ key, label }) => (
                <TouchableOpacity
                  key={key}
                  style={[styles.chip, active === key && styles.chipActive]}
                  onPress={() => { onChange(key); setOpen(false); }}
                >
                  <Text style={[styles.chipText, active === key && styles.chipTextActive]}>
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
  btn: {
    alignSelf: 'flex-start',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#252830',
  },
  btnActive: { backgroundColor: '#4ecdc4' },
  btnText: { color: '#aaa', fontSize: 13, fontWeight: '600' },
  btnTextActive: { color: '#fff' },
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#1a1c23',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
    paddingBottom: 40,
  },
  title: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 16,
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  chip: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: '#252830',
  },
  chipActive: { backgroundColor: '#4ecdc4' },
  chipText: { color: '#aaa', fontSize: 14, fontWeight: '600' },
  chipTextActive: { color: '#fff' },
});
