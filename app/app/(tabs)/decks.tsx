import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  Alert,
  StyleSheet,
  TextInput,
  Modal,
  type ListRenderItem,
} from 'react-native';
import { useRouter } from 'expo-router';
import { memo, useCallback, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getDecks, createDeck, deleteDeck, type Deck } from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';
import { useKeyboardAppearance, useTheme } from '../../src/theme/useTheme';
import { Icon } from '../../src/components/icons/Icon';

const FORMATS = ['Commander', 'Standard', 'Modern', 'Legacy', 'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other'];
const ROW_HEIGHT = 62; // padding 14*2 + content ~22 + marginBottom 8

type Theme = ReturnType<typeof useTheme>;

const keyExtractor = (d: Deck) => String(d.id);
const getItemLayout = (_: ArrayLike<Deck> | null | undefined, index: number) =>
  ({ length: ROW_HEIGHT, offset: ROW_HEIGHT * index, index });

type DeckRowProps = {
  deck: Deck;
  active: boolean;
  theme: Theme;
  onPress: (id: number) => void;
  onLongPress: (id: number, name: string) => void;
};

const DeckRow = memo(function DeckRow({ deck, active, theme, onPress, onLongPress }: DeckRowProps) {
  const handlePress = useCallback(() => onPress(deck.id), [onPress, deck.id]);
  const handleLongPress = useCallback(() => onLongPress(deck.id, deck.name), [onLongPress, deck.id, deck.name]);
  return (
    <TouchableOpacity
      style={[
        styles.deckRow,
        { backgroundColor: theme.surface },
        active && { borderWidth: 1, borderColor: theme.accent },
      ]}
      onPress={handlePress}
      onLongPress={handleLongPress}
    >
      <View>
        <Text style={[styles.deckName, { color: theme.text }]}>{deck.name}</Text>
        <Text style={[styles.deckFormat, { color: theme.textSecondary }]}>{deck.format}</Text>
      </View>
      {active && <Text style={[styles.activeBadge, { color: theme.accent }]}>Active</Text>}
    </TouchableOpacity>
  );
});

export default function DecksScreen() {
  const theme = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const router = useRouter();
  const qc = useQueryClient();
  const activeDeckId = useStore((s) => s.activeDeckId);
  const setActiveDeckId = useStore((s) => s.setActiveDeckId);

  const [modalVisible, setModalVisible] = useState(false);
  const [deckName, setDeckName] = useState('');
  const [selectedFormat, setSelectedFormat] = useState('Commander');

  const { data: decks = [] } = useQuery({ queryKey: ['decks'], queryFn: getDecks });

  const handleCreate = useCallback(() => {
    const trimmed = deckName.trim();
    if (!trimmed) return;
    const id = createDeck({ name: trimmed, format: selectedFormat });
    qc.invalidateQueries({ queryKey: ['decks'] });
    setActiveDeckId(id);
    setModalVisible(false);
    setDeckName('');
    router.push(`/deck/${id}`);
  }, [deckName, selectedFormat, qc, setActiveDeckId, router]);

  const handleLongPress = useCallback(
    (deckId: number, name: string) => {
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
    },
    [qc, activeDeckId, setActiveDeckId],
  );

  const handleSelectDeck = useCallback(
    (id: number) => {
      setActiveDeckId(id);
      router.push(`/deck/${id}`);
    },
    [setActiveDeckId, router],
  );

  const renderItem = useCallback<ListRenderItem<Deck>>(
    ({ item }) => (
      <DeckRow
        deck={item}
        active={item.id === activeDeckId}
        theme={theme}
        onPress={handleSelectDeck}
        onLongPress={handleLongPress}
      />
    ),
    [activeDeckId, theme, handleSelectDeck, handleLongPress],
  );

  const openModal = useCallback(() => setModalVisible(true), []);
  const closeModal = useCallback(() => {
    setModalVisible(false);
    setDeckName('');
  }, []);

  const emptyText = useMemo(
    () => <Text style={[styles.empty, { color: theme.textSecondary }]}>No decks yet. Create one!</Text>,
    [theme.textSecondary],
  );

  const canCreate = deckName.trim().length > 0;

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <TouchableOpacity
        style={[styles.createBtn, { backgroundColor: theme.accent }]}
        onPress={openModal}
        accessibilityRole="button"
        accessibilityLabel="New deck"
      >
        <Icon name="plus" size={18} color={theme.text} strokeWidth={2.5} />
        <Text style={[styles.createBtnText, { color: theme.text }]}>New Deck</Text>
      </TouchableOpacity>

      <FlatList
        data={decks}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.list}
        renderItem={renderItem}
        getItemLayout={getItemLayout}
        initialNumToRender={12}
        maxToRenderPerBatch={12}
        windowSize={7}
        ListEmptyComponent={emptyText}
      />

      <Modal visible={modalVisible} transparent animationType="slide" onRequestClose={closeModal}>
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
              keyboardAppearance={keyboardAppearance}
            />
            <Text style={[styles.modalLabel, { color: theme.textSecondary }]}>Format</Text>
            <View style={styles.formatGrid}>
              {FORMATS.map((f) => {
                const isActive = selectedFormat === f;
                return (
                  <TouchableOpacity
                    key={f}
                    style={[styles.formatChip, { backgroundColor: isActive ? theme.accent : theme.surfaceAlt }]}
                    onPress={() => setSelectedFormat(f)}
                  >
                    <Text
                      style={[
                        styles.formatChipText,
                        { color: isActive ? theme.text : theme.textSecondary },
                        isActive && styles.formatChipTextActive,
                      ]}
                    >
                      {f}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            <View style={styles.modalBtns}>
              <TouchableOpacity style={[styles.modalBtn, { backgroundColor: theme.surfaceAlt }]} onPress={closeModal}>
                <Text style={[styles.modalBtnText, { color: theme.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalBtn, { backgroundColor: theme.accent }, !canCreate && styles.modalBtnDisabled]}
                onPress={handleCreate}
                disabled={!canCreate}
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
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
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
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', alignItems: 'center', justifyContent: 'flex-end' },
  modalBox: {
    width: '100%',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
    gap: 12,
  },
  modalTitle: { fontSize: 18, fontWeight: '700' },
  modalInput: { borderRadius: 8, padding: 12, fontSize: 15 },
  modalLabel: { fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 },
  formatGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center' },
  formatChip: {
    minWidth: 80,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  formatChipText: { fontSize: 12 },
  formatChipTextActive: { fontWeight: '600' },
  modalBtns: { flexDirection: 'row', gap: 8, marginTop: 4 },
  modalBtn: { flex: 1, borderRadius: 8, padding: 12, alignItems: 'center' },
  modalBtnDisabled: { opacity: 0.4 },
  modalBtnText: { fontWeight: '700' },
});
