import {
  View,
  TextInput,
  FlatList,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import { useState } from 'react';
import { CardRow } from '../../src/components/CardRow';
import { useScryfallSearch, useSynergySearch } from '../../src/api/hooks';
import { useTheme } from '../../src/theme/useTheme';

type Mode = 'search' | 'synergy';

export default function SearchScreen() {
  const theme = useTheme();
  const [query, setQuery] = useState('');
  const [synergyCard, setSynergyCard] = useState('');
  const [mode, setMode] = useState<Mode>('search');

  const searchResults = useScryfallSearch(mode === 'search' ? query : '');
  const synergyResults = useSynergySearch(mode === 'synergy' ? synergyCard : '');

  const active = mode === 'search' ? searchResults : synergyResults;
  const results = active.data ?? [];

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <View style={styles.modeSwitcher}>
        <TouchableOpacity
          onPress={() => setMode('search')}
          style={[styles.modeBtn, { backgroundColor: mode === 'search' ? theme.accent : theme.surface }]}
        >
          <Text style={[styles.modeBtnText, { color: mode === 'search' ? theme.text : theme.textSecondary }]}>
            Search
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setMode('synergy')}
          style={[styles.modeBtn, { backgroundColor: mode === 'synergy' ? theme.accent : theme.surface }]}
        >
          <Text style={[styles.modeBtnText, { color: mode === 'synergy' ? theme.text : theme.textSecondary }]}>
            ⚡ Synergy
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.inputRow}>
        {mode === 'search' ? (
          <TextInput
            style={[styles.input, { backgroundColor: theme.surface, color: theme.text }]}
            value={query}
            onChangeText={setQuery}
            placeholder='Search cards (e.g. "draws a card")'
            placeholderTextColor={theme.textSecondary}
            autoCorrect={false}
          />
        ) : (
          <TextInput
            style={[styles.input, { backgroundColor: theme.surface, color: theme.text }]}
            value={synergyCard}
            onChangeText={setSynergyCard}
            placeholder="Enter a card name (e.g. Teysa Karlov)"
            placeholderTextColor={theme.textSecondary}
            autoCorrect={false}
          />
        )}
      </View>

      {active.isLoading && (
        <ActivityIndicator style={styles.loader} color={theme.accent} />
      )}

      {mode === 'synergy' && synergyCard.length > 1 && !active.isLoading && (
        <Text style={[styles.synergyHint, { color: theme.textSecondary }]}>
          Showing cards that synergize with {synergyCard}
        </Text>
      )}

      <FlatList
        data={results}
        keyExtractor={(c) => c.scryfall_id}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => <CardRow card={item} />}
        ListEmptyComponent={
          !active.isLoading && (query.length > 1 || synergyCard.length > 1) ? (
            <Text style={[styles.empty, { color: theme.textSecondary }]}>No results</Text>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  modeSwitcher: { flexDirection: 'row', padding: 12, gap: 8 },
  modeBtn: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  modeBtnText: { fontWeight: '600' },
  inputRow: { paddingHorizontal: 12, paddingBottom: 8 },
  input: {
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
  },
  loader: { marginTop: 20 },
  synergyHint: { fontSize: 12, paddingHorizontal: 16, marginBottom: 8 },
  list: { padding: 12 },
  empty: { textAlign: 'center', marginTop: 40 },
});
