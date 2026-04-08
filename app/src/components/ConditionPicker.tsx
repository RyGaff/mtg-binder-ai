import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';

export type Condition = 'NM' | 'LP' | 'MP' | 'HP' | 'DMG';
const CONDITIONS: Condition[] = ['NM', 'LP', 'MP', 'HP', 'DMG'];

type Props = { value: Condition; onChange: (c: Condition) => void };

export function ConditionPicker({ value, onChange }: Props) {
  return (
    <View style={styles.row}>
      {CONDITIONS.map((c) => (
        <TouchableOpacity
          key={c}
          onPress={() => onChange(c)}
          style={[styles.chip, value === c && styles.active]}
        >
          <Text style={[styles.label, value === c && styles.activeLabel]}>{c}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', gap: 6 },
  chip: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 8, backgroundColor: '#1a1c23' },
  active: { backgroundColor: '#4ecdc4' },
  label: { color: '#aaa', fontSize: 12 },
  activeLabel: { color: '#fff', fontWeight: '700' },
});
