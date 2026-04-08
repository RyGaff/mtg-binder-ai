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

type Mode = 'search' | 'synergy';

export default function SearchScreen() {
  const [query, setQuery] = useState('');
  const [synergyCard, setSynergyCard] = useState('');
  const [mode, setMode] = useState<Mode>('search');

  const searchResults = useScryfallSearch(mode === 'search' ? query : '');
  const synergyResults = useSynergySearch(mode === 'synergy' ? synergyCard : '');

  const active = mode === 'search' ? searchResults : synergyResults;
  const results = active.data ?? [];

  return (
    <View style={styles.screen}>
      <View style={styles.modeSwitcher}>
        <TouchableOpacity
          onPress={() => setMode('search')}
          style={[styles.modeBtn, mode === 'search' && styles.modeBtnActive]}
        >
          <Text style={[styles.modeBtnText, mode === 'search' && styles.modeBtnTextActive]}>
            Search
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setMode('synergy')}
          style={[styles.modeBtn, mode === 'synergy' && styles.modeBtnActive]}
        >
          <Text style={[styles.modeBtnText, mode === 'synergy' && styles.modeBtnTextActive]}>
            ⚡ Synergy
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.inputRow}>
        {mode === 'search' ? (
          <TextInput
            style={styles.input}
            value={query}
            onChangeText={setQuery}
            placeholder='Search cards (e.g. "draws a card")'
            placeholderTextColor="#555"
            autoCorrect={false}
          />
        ) : (
          <TextInput
            style={styles.input}
            value={synergyCard}
            onChangeText={setSynergyCard}
            placeholder="Enter a card name (e.g. Teysa Karlov)"
            placeholderTextColor="#555"
            autoCorrect={false}
          />
        )}
      </View>

      {active.isLoading && (
        <ActivityIndicator style={styles.loader} color="#4ecdc4" />
      )}

      {mode === 'synergy' && synergyCard.length > 1 && !active.isLoading && (
        <Text style={styles.synergyHint}>
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
            <Text style={styles.empty}>No results</Text>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#111318' },
  modeSwitcher: { flexDirection: 'row', padding: 12, gap: 8 },
  modeBtn: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    backgroundColor: '#1a1c23',
    alignItems: 'center',
  },
  modeBtnActive: { backgroundColor: '#4ecdc4' },
  modeBtnText: { color: '#888', fontWeight: '600' },
  modeBtnTextActive: { color: '#fff' },
  inputRow: { paddingHorizontal: 12, paddingBottom: 8 },
  input: {
    backgroundColor: '#1a1c23',
    color: '#fff',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
  },
  loader: { marginTop: 20 },
  synergyHint: { color: '#888', fontSize: 12, paddingHorizontal: 16, marginBottom: 8 },
  list: { padding: 12 },
  empty: { color: '#555', textAlign: 'center', marginTop: 40 },
});
