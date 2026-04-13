import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  Alert,
  StyleSheet,
  TextInput,
  Modal,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getDecks, createDeck, deleteDeck } from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';

const FORMATS = [
  'Commander', 'Standard', 'Modern', 'Legacy',
  'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other',
];

export default function DecksScreen() {
  const theme = useTheme();
  const router = useRouter();
  const qc = useQueryClient();
  const { activeDeckId, setActiveDeckId } = useStore();

  const [modalVisible, setModalVisible] = useState(false);
  const [deckName, setDeckName] = useState('');
  const [selectedFormat, setSelectedFormat] = useState('Commander');

  const { data: decks = [] } = useQuery({
    queryKey: ['decks'],
    queryFn: getDecks,
  });

  const handleCreate = () => {
    if (!deckName.trim()) return;
    const id = createDeck({ name: deckName.trim(), format: selectedFormat });
    qc.invalidateQueries({ queryKey: ['decks'] });
    setActiveDeckId(id);
    setModalVisible(false);
    setDeckName('');
    router.push(`/deck/${id}`);
  };

  const handleLongPress = (deckId: number, name: string) => {
    Alert.alert('Delete Deck', `Delete "${name}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          deleteDeck(deckId);
          qc.invalidateQueries({ queryKey: ['decks'] });
          if (activeDeckId === deckId) setActiveDeckId(null);
        },
      },
    ]);
  };

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <TouchableOpacity
        style={[styles.createBtn, { backgroundColor: theme.accent }]}
        onPress={() => setModalVisible(true)}
      >
        <Text style={[styles.createBtnText, { color: theme.text }]}>+ New Deck</Text>
      </TouchableOpacity>

      <FlatList
        data={decks}
        keyExtractor={(d) => String(d.id)}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[
              styles.deckRow,
              { backgroundColor: theme.surface },
              item.id === activeDeckId && { borderWidth: 1, borderColor: theme.accent },
            ]}
            onPress={() => {
              setActiveDeckId(item.id);
              router.push(`/deck/${item.id}`);
            }}
            onLongPress={() => handleLongPress(item.id, item.name)}
          >
            <View>
              <Text style={[styles.deckName, { color: theme.text }]}>{item.name}</Text>
              <Text style={[styles.deckFormat, { color: theme.textSecondary }]}>{item.format}</Text>
            </View>
            {item.id === activeDeckId && (
              <Text style={[styles.activeBadge, { color: theme.accent }]}>Active</Text>
            )}
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <Text style={[styles.empty, { color: theme.textSecondary }]}>No decks yet. Create one!</Text>
        }
      />

      <Modal
        visible={modalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalBox, { backgroundColor: theme.surface }]}>
            <Text style={[styles.modalTitle, { color: theme.text }]}>New Deck</Text>
            <TextInput
              style={[styles.modalInput, { backgroundColor: theme.bg, color: theme.text }]}
              value={deckName}
              onChangeText={setDeckName}
              placeholder="Deck name"
              placeholderTextColor={theme.textSecondary}
              autoFocus
            />
            <Text style={[styles.modalLabel, { color: theme.textSecondary }]}>Format</Text>
            <View style={styles.formatGrid}>
              {FORMATS.map((f) => (
                <TouchableOpacity
                  key={f}
                  style={[
                    styles.formatChip,
                    { backgroundColor: selectedFormat === f ? theme.accent : theme.surfaceAlt },
                  ]}
                  onPress={() => setSelectedFormat(f)}
                >
                  <Text
                    style={[
                      styles.formatChipText,
                      { color: selectedFormat === f ? theme.text : theme.textSecondary },
                      selectedFormat === f && styles.formatChipTextActive,
                    ]}
                  >
                    {f}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            <View style={styles.modalBtns}>
              <TouchableOpacity
                style={[styles.modalBtn, { backgroundColor: theme.surfaceAlt }]}
                onPress={() => { setModalVisible(false); setDeckName(''); }}
              >
                <Text style={[styles.modalBtnText, { color: theme.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalBtn, { backgroundColor: theme.accent }, !deckName.trim() && styles.modalBtnDisabled]}
                onPress={handleCreate}
                disabled={!deckName.trim()}
              >
                <Text style={[styles.modalBtnText, { color: theme.text }]}>Create</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  createBtn: {
    margin: 16,
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  createBtnText: { fontWeight: '700', fontSize: 15 },
  list: { paddingHorizontal: 16 },
  deckRow: {
    borderRadius: 8,
    padding: 14,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  deckName: { fontWeight: '600', fontSize: 15 },
  deckFormat: { fontSize: 12, marginTop: 2 },
  activeBadge: { fontSize: 11, fontWeight: '700' },
  empty: { textAlign: 'center', marginTop: 60 },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  modalBox: {
    width: '100%',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
    gap: 12,
  },
  modalTitle: { fontSize: 18, fontWeight: '700' },
  modalInput: {
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
  },
  modalLabel: { fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 },
  formatGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  formatChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  formatChipText: { fontSize: 12 },
  formatChipTextActive: { fontWeight: '600' },
  modalBtns: { flexDirection: 'row', gap: 8, marginTop: 4 },
  modalBtn: {
    flex: 1,
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  modalBtnDisabled: { opacity: 0.4 },
  modalBtnText: { fontWeight: '700' },
});
