import {
  View, Text, FlatList, Pressable, Modal, TextInput, ScrollView,
  StyleSheet, KeyboardAvoidingView,
  type ListRenderItem,
} from 'react-native';
import { useCallback, useMemo, useState } from 'react';
import { useRouter } from 'expo-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import {
  getDecksWithMeta, createDeck, deleteDeck, renameDeck, exportDeckAsText,
  type DeckWithMeta,
} from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';
import { useKeyboardAppearance, useTheme } from '../../src/theme/useTheme';
import { Icon } from '../../src/components/icons/Icon';
import { DeckRow } from '../../src/components/DeckRow';

const FORMATS = ['Commander', 'Standard', 'Modern', 'Legacy', 'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other'] as const;
type Format = typeof FORMATS[number];

type SheetAction = { label: string; destructive?: boolean; onPress: () => void };
type Sheet = { title: string; subtitle?: string; actions: SheetAction[] } | null;

const FILTER_OPTIONS: ('All' | Format)[] = ['All', ...FORMATS];

const press = (extra: any) => ({ pressed }: { pressed: boolean }) =>
  [extra, pressed && { opacity: 0.7 }];

export default function DecksScreen() {
  const t = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const router = useRouter();
  const qc = useQueryClient();

  const activeDeckId = useStore((s) => s.activeDeckId);
  const setActiveDeckId = useStore((s) => s.setActiveDeckId);
  const mode = useStore((s) => s.deckListMode);
  const setMode = useStore((s) => s.setDeckListMode);

  const { data: decks = [] } = useQuery({ queryKey: ['decks'], queryFn: getDecksWithMeta });

  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<'All' | Format>('All');
  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [format, setFormat] = useState<Format>('Commander');
  const [sheet, setSheet] = useState<Sheet>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return decks.filter((d) =>
      (filter === 'All' || d.format === filter) &&
      (!q || d.name.toLowerCase().includes(q) || d.format.toLowerCase().includes(q))
    );
  }, [decks, query, filter]);

  const refresh = useCallback(() => {
    qc.invalidateQueries({ queryKey: ['decks'] });
  }, [qc]);

  const submit = useCallback(() => {
    const trimmed = name.trim();
    if (!trimmed) return;
    if (editingId) {
      renameDeck(editingId, trimmed);
    } else {
      const id = createDeck({ name: trimmed, format });
      setActiveDeckId(id);
    }
    refresh();
    setModalOpen(false);
    setName('');
  }, [name, format, editingId, setActiveDeckId, refresh]);

  const handleSelect = useCallback((id: number) => {
    setActiveDeckId(id);
    router.push(`/deck/${id}`);
  }, [setActiveDeckId, router]);

  const exportDeck = useCallback(async (deck: DeckWithMeta) => {
    try {
      const text = exportDeckAsText(deck.id);
      const path = `${FileSystem.cacheDirectory}${deck.name}.txt`;
      await FileSystem.writeAsStringAsync(path, text, { encoding: FileSystem.EncodingType.UTF8 });
      await Sharing.shareAsync(path, { mimeType: 'text/plain' });
    } catch (e) {
      console.warn('exportDeck failed', e);
    }
  }, []);

  const removeDeck = useCallback((deck: DeckWithMeta) => {
    setSheet({
      title: 'Delete deck?',
      subtitle: `"${deck.name}" will be removed.`,
      actions: [{
        label: 'Delete',
        destructive: true,
        onPress: () => {
          deleteDeck(deck.id);
          if (activeDeckId === deck.id) setActiveDeckId(null);
          refresh();
        },
      }],
    });
  }, [activeDeckId, setActiveDeckId, refresh]);

  const openOptions = useCallback((deck: DeckWithMeta) => {
    setSheet({
      title: deck.name,
      subtitle: `${deck.format} · ${deck.card_count} cards`,
      actions: [
        { label: 'Rename', onPress: () => { setEditingId(deck.id); setName(deck.name); setModalOpen(true); } },
        { label: 'Export', onPress: () => { void exportDeck(deck); } },
        { label: 'Delete', destructive: true, onPress: () => removeDeck(deck) },
      ],
    });
  }, [exportDeck, removeDeck]);

  const renderItem = useCallback<ListRenderItem<DeckWithMeta>>(({ item }) => (
    <DeckRow
      deck={item}
      mode={mode}
      active={item.id === activeDeckId}
      onPress={handleSelect}
      onLongPress={openOptions}
      onMore={openOptions}
    />
  ), [mode, activeDeckId, handleSelect, openOptions]);

  const openCreate = () => { setEditingId(null); setName(''); setFormat('Commander'); setModalOpen(true); };
  const closeModal = () => { setModalOpen(false); setName(''); };

  return (
    <View style={[s.screen, { backgroundColor: t.bg }]}>
      <View style={s.topRow}>
        <View style={[s.searchWrap, { backgroundColor: t.surface, borderColor: t.border }]}>
          <TextInput
            value={query} onChangeText={setQuery}
            placeholder="Search decks" placeholderTextColor={t.textSecondary}
            style={[s.searchInput, { color: t.text }]}
            autoCorrect={false} autoCapitalize="none"
            keyboardAppearance={keyboardAppearance}
          />
          {query.length > 0 ? (
            <Pressable onPress={() => setQuery('')} hitSlop={8}>
              <Text style={[s.clear, { color: t.textSecondary }]}>×</Text>
            </Pressable>
          ) : null}
        </View>
        <Pressable
          onPress={() => setMode(mode === 'banner' ? 'compact' : 'banner')}
          hitSlop={6}
          style={press([s.viewToggle, { borderColor: t.border, backgroundColor: t.surface }])}
          accessibilityLabel={`Switch to ${mode === 'banner' ? 'compact' : 'banner'} view`}
        >
          <Text style={[s.viewToggleText, { color: t.text }]}>{mode === 'banner' ? '☰' : '▢'}</Text>
        </Pressable>
      </View>

      <ScrollView
        horizontal showsHorizontalScrollIndicator={false}
        style={s.chipsScroll} contentContainerStyle={s.chipsContent}
      >
        {FILTER_OPTIONS.map((f) => {
          const active = f === filter;
          return (
            <Pressable
              key={f} onPress={() => setFilter(f)}
              style={[s.chip, { backgroundColor: active ? t.accent : t.surfaceAlt }]}
            >
              <Text style={[s.chipText, { color: active ? t.text : t.textSecondary }, active && s.chipTextActive]}>{f}</Text>
            </Pressable>
          );
        })}
      </ScrollView>

      <FlatList
        data={filtered}
        keyExtractor={(d) => String(d.id)}
        contentContainerStyle={[s.list, !filtered.length && s.emptyWrap]}
        renderItem={renderItem}
        ItemSeparatorComponent={mode === 'banner' ? () => <View style={{ height: 10 }} /> : null}
        ListEmptyComponent={
          <View style={s.empty}>
            <Text style={[s.emptyTitle, { color: t.text }]}>
              {decks.length === 0 ? 'No decks yet' : 'No matches'}
            </Text>
            <Text style={[s.emptyBody, { color: t.textSecondary }]}>
              {decks.length === 0 ? 'Tap + to create your first deck.' : 'Try a different name or format.'}
            </Text>
          </View>
        }
      />

      <Pressable
        onPress={openCreate}
        accessibilityRole="button"
        accessibilityLabel="New deck"
        style={({ pressed }) => [s.fab, { backgroundColor: t.accent },
          pressed && { opacity: 0.8, transform: [{ scale: 0.96 }] }]}
      >
        <Icon name="plus" size={26} color={t.text} strokeWidth={2.5} />
      </Pressable>

      <Modal visible={modalOpen} transparent animationType="slide" onRequestClose={closeModal}>
        <Pressable style={s.backdrop} onPress={closeModal}>
          <KeyboardAvoidingView behavior="padding" style={s.modalAnchor}>
            <Pressable onPress={(e) => e.stopPropagation()} style={[s.modal, { backgroundColor: t.surface }]}>
              <Text style={[s.modalTitle, { color: t.text }]}>{editingId ? 'Rename Deck' : 'New Deck'}</Text>
              <TextInput
                value={name} onChangeText={setName}
                placeholder="Deck name" placeholderTextColor={t.textSecondary}
                style={[s.modalInput, { backgroundColor: t.bg, color: t.text }]}
                autoFocus selectTextOnFocus returnKeyType="done"
                keyboardAppearance={keyboardAppearance}
                onSubmitEditing={submit}
              />
              {!editingId ? (
                <>
                  <Text style={[s.modalLabel, { color: t.textSecondary }]}>Format</Text>
                  <View style={s.formatGrid}>
                    {FORMATS.map((f) => {
                      const active = f === format;
                      return (
                        <Pressable
                          key={f} onPress={() => setFormat(f)}
                          style={[s.formatChip, { backgroundColor: active ? t.accent : t.surfaceAlt }]}
                        >
                          <Text style={[s.formatChipText, { color: active ? t.text : t.textSecondary }, active && s.chipTextActive]}>{f}</Text>
                        </Pressable>
                      );
                    })}
                  </View>
                </>
              ) : null}
              <View style={s.modalBtns}>
                <Pressable onPress={closeModal} style={press([s.modalBtn, { backgroundColor: t.surfaceAlt }])}>
                  <Text style={[s.modalBtnText, { color: t.text }]}>Cancel</Text>
                </Pressable>
                <Pressable
                  onPress={submit}
                  disabled={!name.trim()}
                  style={({ pressed }) => [s.modalBtn, { backgroundColor: t.accent },
                    !name.trim() && { opacity: 0.4 }, pressed && { opacity: 0.8 }]}
                >
                  <Text style={[s.modalBtnText, { color: t.text }]}>{editingId ? 'Save' : 'Create'}</Text>
                </Pressable>
              </View>
            </Pressable>
          </KeyboardAvoidingView>
        </Pressable>
      </Modal>

      <Modal visible={!!sheet} transparent animationType="fade" onRequestClose={() => setSheet(null)}>
        <Pressable style={s.backdrop} onPress={() => setSheet(null)}>
          <View style={s.modalCenter}>
            <Pressable onPress={(e) => e.stopPropagation()} style={[s.modal, { backgroundColor: t.surface }]}>
              {sheet ? (
                <>
                  <Text style={[s.modalTitle, { color: t.text }]}>{sheet.title}</Text>
                  {sheet.subtitle ? <Text style={[s.sheetSubtitle, { color: t.textSecondary }]}>{sheet.subtitle}</Text> : null}
                  <View style={s.sheetActions}>
                    {sheet.actions.map((a, i) => (
                      <Pressable
                        key={i}
                        onPress={() => { setSheet(null); a.onPress(); }}
                        style={press([s.sheetBtn, { backgroundColor: a.destructive ? t.danger : t.accent }])}
                      >
                        <Text style={[s.modalBtnText, { color: t.text }]}>{a.label}</Text>
                      </Pressable>
                    ))}
                  </View>
                </>
              ) : null}
            </Pressable>
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

const s = StyleSheet.create({
  screen: { flex: 1 },
  topRow: { flexDirection: 'row', gap: 8, paddingHorizontal: 16, paddingTop: 10, paddingBottom: 6 },
  searchWrap: { flex: 1, flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, height: 40, borderRadius: 10, borderWidth: 1, gap: 6 },
  searchInput: { flex: 1, fontSize: 14, height: '100%' },
  clear: { fontSize: 18, paddingHorizontal: 4 },
  viewToggle: { width: 40, height: 40, borderRadius: 10, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  viewToggleText: { fontSize: 16 },
  chipsScroll: { maxHeight: 40, flexGrow: 0 },
  chipsContent: { paddingHorizontal: 16, gap: 6, alignItems: 'center', paddingBottom: 6 },
  chip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999 },
  chipText: { fontSize: 12 },
  list: { paddingHorizontal: 16, paddingBottom: 100, paddingTop: 4 },
  emptyWrap: { flexGrow: 1, alignItems: 'center', justifyContent: 'center' },
  empty: { alignItems: 'center', gap: 6, paddingHorizontal: 32 },
  emptyTitle: { fontSize: 18, fontWeight: '700' },
  emptyBody: { fontSize: 14, textAlign: 'center' },
  fab: {
    position: 'absolute', right: 24, bottom: 32, width: 56, height: 56, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center', elevation: 6,
    shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 6, shadowOffset: { width: 0, height: 3 },
  },
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)' },
  modalAnchor: { flex: 1, justifyContent: 'flex-end' },
  modalCenter: { flex: 1, justifyContent: 'center', paddingHorizontal: 24 },
  modal: { borderTopLeftRadius: 16, borderTopRightRadius: 16, borderRadius: 16, padding: 20, gap: 10 },
  modalTitle: { fontSize: 18, fontWeight: '700' },
  modalInput: { borderRadius: 8, padding: 12, fontSize: 15 },
  modalLabel: { fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginTop: 4 },
  formatGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  formatChip: { minWidth: 80, paddingHorizontal: 12, paddingVertical: 10, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  formatChipText: { fontSize: 12 },
  modalBtns: { flexDirection: 'row', gap: 8, marginTop: 8 },
  modalBtn: { flex: 1, borderRadius: 8, padding: 12, alignItems: 'center' },
  modalBtnText: { fontWeight: '700' },
  sheetSubtitle: { fontSize: 13, marginTop: -2, marginBottom: 4 },
  sheetActions: { gap: 8, marginTop: 8 },
  sheetBtn: { width: '100%', borderRadius: 10, paddingVertical: 14, alignItems: 'center' },
});
