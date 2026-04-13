import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useTheme } from '../theme/useTheme';

export type Condition = 'NM' | 'LP' | 'MP' | 'HP' | 'DMG';
const CONDITIONS: Condition[] = ['NM', 'LP', 'MP', 'HP', 'DMG'];

type Props = { value: Condition; onChange: (c: Condition) => void };

export function ConditionPicker({ value, onChange }: Props) {
  const theme = useTheme();
  return (
    <View style={styles.row}>
      {CONDITIONS.map((c) => (
        <TouchableOpacity
          key={c}
          onPress={() => onChange(c)}
          style={[styles.chip, { backgroundColor: value === c ? theme.accent : theme.surface }]}
        >
          <Text style={[styles.label, { color: value === c ? theme.text : theme.textSecondary, fontWeight: value === c ? '700' : '400' }]}>
            {c}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', gap: 6 },
  chip: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 8 },
  label: { fontSize: 12 },
});
