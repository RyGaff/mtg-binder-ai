import {
  View,
  TextInput,
  FlatList,
  Text,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import { useState } from 'react';
import { CardRow } from '../../src/components/CardRow';
import { useScryfallSearch } from '../../src/api/hooks';
import { useTheme } from '../../src/theme/useTheme';

export default function SearchScreen() {
  const theme = useTheme();
  const [query, setQuery] = useState('');
  const { data: results = [], isLoading } = useScryfallSearch(query);

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <View style={styles.inputRow}>
        <TextInput
          style={[styles.input, { backgroundColor: theme.surface, color: theme.text }]}
          value={query}
          onChangeText={setQuery}
          placeholder='Search cards (e.g. "draws a card")'
          placeholderTextColor={theme.textSecondary}
          autoCorrect={false}
        />
      </View>

      {isLoading && <ActivityIndicator style={styles.loader} color={theme.accent} />}

      <FlatList
        data={results}
        keyExtractor={(c) => c.scryfall_id}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => <CardRow card={item} />}
        ListEmptyComponent={
          !isLoading && query.length > 1 ? (
            <Text style={[styles.empty, { color: theme.textSecondary }]}>No results</Text>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  inputRow: { padding: 12 },
  input: {
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
  },
  loader: { marginTop: 20 },
  list: { padding: 12 },
  empty: { textAlign: 'center', marginTop: 40 },
});
