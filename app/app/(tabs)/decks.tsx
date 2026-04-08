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

const FORMATS = [
  'Commander', 'Standard', 'Modern', 'Legacy',
  'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other',
];

export default function DecksScreen() {
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
    <View style={styles.screen}>
      <TouchableOpacity
        style={styles.createBtn}
        onPress={() => setModalVisible(true)}
      >
        <Text style={styles.createBtnText}>+ New Deck</Text>
      </TouchableOpacity>

      <FlatList
        data={decks}
        keyExtractor={(d) => String(d.id)}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[styles.deckRow, item.id === activeDeckId && styles.deckRowActive]}
            onPress={() => {
              setActiveDeckId(item.id);
              router.push(`/deck/${item.id}`);
            }}
            onLongPress={() => handleLongPress(item.id, item.name)}
          >
            <View>
              <Text style={styles.deckName}>{item.name}</Text>
              <Text style={styles.deckFormat}>{item.format}</Text>
            </View>
            {item.id === activeDeckId && (
              <Text style={styles.activeBadge}>Active</Text>
            )}
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <Text style={styles.empty}>No decks yet. Create one!</Text>
        }
      />

      <Modal
        visible={modalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalBox}>
            <Text style={styles.modalTitle}>New Deck</Text>
            <TextInput
              style={styles.modalInput}
              value={deckName}
              onChangeText={setDeckName}
              placeholder="Deck name"
              placeholderTextColor="#555"
              autoFocus
            />
            <Text style={styles.modalLabel}>Format</Text>
            <View style={styles.formatGrid}>
              {FORMATS.map((f) => (
                <TouchableOpacity
                  key={f}
                  style={[
                    styles.formatChip,
                    selectedFormat === f && styles.formatChipActive,
                  ]}
                  onPress={() => setSelectedFormat(f)}
                >
                  <Text
                    style={[
                      styles.formatChipText,
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
                style={[styles.modalBtn, styles.modalBtnCancel]}
                onPress={() => { setModalVisible(false); setDeckName(''); }}
              >
                <Text style={styles.modalBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalBtn, !deckName.trim() && styles.modalBtnDisabled]}
                onPress={handleCreate}
                disabled={!deckName.trim()}
              >
                <Text style={styles.modalBtnText}>Create</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#111318' },
  createBtn: {
    margin: 16,
    backgroundColor: '#4ecdc4',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  createBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  list: { paddingHorizontal: 16 },
  deckRow: {
    backgroundColor: '#1a1c23',
    borderRadius: 8,
    padding: 14,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  deckRowActive: { borderWidth: 1, borderColor: '#4ecdc4' },
  deckName: { color: '#fff', fontWeight: '600', fontSize: 15 },
  deckFormat: { color: '#888', fontSize: 12, marginTop: 2 },
  activeBadge: { color: '#4ecdc4', fontSize: 11, fontWeight: '700' },
  empty: { color: '#555', textAlign: 'center', marginTop: 60 },
  modalOverlay: {
    flex: 1,
    backgroundColor: '#000a',
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  modalBox: {
    width: '100%',
    backgroundColor: '#1a1c23',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
    gap: 12,
  },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },
  modalInput: {
    backgroundColor: '#0f0f1a',
    color: '#fff',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
  },
  modalLabel: { color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 },
  formatGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  formatChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: '#0f0f1a',
  },
  formatChipActive: { backgroundColor: '#4ecdc4' },
  formatChipText: { color: '#aaa', fontSize: 12 },
  formatChipTextActive: { color: '#fff', fontWeight: '600' },
  modalBtns: { flexDirection: 'row', gap: 8, marginTop: 4 },
  modalBtn: {
    flex: 1,
    backgroundColor: '#4ecdc4',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  modalBtnCancel: { backgroundColor: '#252830' },
  modalBtnDisabled: { opacity: 0.4 },
  modalBtnText: { color: '#fff', fontWeight: '700' },
});
