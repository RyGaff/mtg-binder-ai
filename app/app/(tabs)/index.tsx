import {
  View,
  Text,
  TextInput,
  Modal,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
  StyleSheet,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useCallback, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { ColorFilter } from '../../src/components/ColorFilter';
import { CardGrid } from '../../src/components/CardGrid';
import { useStore } from '../../src/store/useStore';
import {
  getCollection,
  getCollectionByColor,
  searchCollection,
  getCollectionTotalValue,
  addManyToCollection,
  clearCollection,
  type AddToCollectionArgs,
} from '../../src/db/collection';
import { upsertCards, getCardBySetNumber, getCardById, type CachedCard } from '../../src/db/cards';
import { serializeToJson, serializeToCsv, parseImportFile } from '../../src/export/collection';
import { fetchCardByName, fetchCardBySetNumber } from '../../src/api/scryfall';
import { useKeyboardAppearance, useTheme } from '../../src/theme/useTheme';
import { spacing, radius, font, MIN_TOUCH, HIT_SLOP_8 } from '../../src/theme/themes';
import { Icon } from '../../src/components/icons/Icon';
import { useActionSheet } from '../../src/components/ActionSheet';
import { useDebouncedValue } from '../../src/hooks/useDebouncedValue';

type ImportProgress = { current: number; total: number; currentName: string };

export default function BinderScreen() {
  const theme = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const router = useRouter();
  const qc = useQueryClient();
  const sheet = useActionSheet();
  const colorFilter = useStore((s) => s.colorFilter);
  const setColorFilter = useStore((s) => s.setColorFilter);

  const [searchQuery, setSearchQuery] = useState('');
  const debouncedSearchQuery = useDebouncedValue(searchQuery, 250);
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null);

  const { data: entries = [] } = useQuery({
    queryKey: ['collection', colorFilter, debouncedSearchQuery],
    queryFn: () => {
      const q = debouncedSearchQuery.trim();
      if (q) return searchCollection(q);
      return colorFilter === 'all' ? getCollection() : getCollectionByColor(colorFilter);
    },
  });

  const { data: totalValue = 0 } = useQuery({
    queryKey: ['collection-value'],
    queryFn: getCollectionTotalValue,
  });

  const exportAs = async (format: 'json' | 'csv') => {
    try {
      const content = format === 'json' ? serializeToJson(entries) : serializeToCsv(entries);
      const path = `${FileSystem.cacheDirectory}collection.${format}`;
      await FileSystem.writeAsStringAsync(path, content, { encoding: FileSystem.EncodingType.UTF8 });
      await Sharing.shareAsync(path);
    } catch (e) {
      console.warn('exportCollection failed', e);
    }
  };

  const handleExport = () => {
    sheet.show({
      title: 'Export Format',
      subtitle: 'Choose a format',
      actions: [
        { label: 'JSON', onPress: () => { void exportAs('json'); } },
        { label: 'CSV', onPress: () => { void exportAs('csv'); } },
      ],
    });
  };

  const handleImport = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ['application/json', 'text/csv', 'text/plain'],
    });
    if (result.canceled) return;
    const asset = result.assets[0];
    const content = await FileSystem.readAsStringAsync(asset.uri);
    const format = asset.name.endsWith('.csv') ? 'csv' : 'json';
    const rows = parseImportFile(content, format);

    const toUpsert: CachedCard[] = [];
    const toAdd: AddToCollectionArgs[] = [];
    let added = 0;
    let failed = 0;
    setImportProgress({ current: 0, total: rows.length, currentName: '' });

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      setImportProgress({ current: i + 1, total: rows.length, currentName: row.name });
      try {
        let scryfallId = row.scryfall_id;
        if (!scryfallId) {
          const cached = row.set_code && row.collector_number
            ? getCardBySetNumber(row.set_code, row.collector_number)
            : null;
          if (cached) {
            scryfallId = cached.scryfall_id;
          } else {
            // Throttle: 100ms between network calls to stay under Scryfall rate limit
            await new Promise((r) => setTimeout(r, 100));
            const card = row.set_code && row.collector_number
              ? await fetchCardBySetNumber(row.set_code, row.collector_number)
              : await fetchCardByName(row.name);
            toUpsert.push(card);
            scryfallId = card.scryfall_id;
          }
        } else if (!getCardById(scryfallId)) {
          await new Promise((r) => setTimeout(r, 100));
          const card = await fetchCardBySetNumber(row.set_code ?? '', row.collector_number ?? '')
            .catch(() => fetchCardByName(row.name));
          toUpsert.push(card);
        }
        toAdd.push({
          scryfall_id: scryfallId,
          quantity: row.quantity,
          foil: row.foil,
          condition: (row.condition as 'NM') ?? 'NM',
        });
        added++;
      } catch {
        failed++;
      }
    }

    if (toUpsert.length) upsertCards(toUpsert);
    if (toAdd.length) addManyToCollection(toAdd);

    setImportProgress(null);
    qc.invalidateQueries({ queryKey: ['collection'] });
    qc.invalidateQueries({ queryKey: ['collection-value'] });
    sheet.show({
      title: 'Import Complete',
      subtitle: failed > 0 ? `Added ${added} cards. ${failed} could not be found.` : `Added ${added} cards.`,
      actions: [],
    });
  };

  const progress = importProgress ? importProgress.current / importProgress.total : 0;
  const handleAddPress = useCallback(() => router.push('/search'), [router]);

  const handleClear = () =>
    sheet.show({
      title: 'Clear Collection',
      subtitle: 'Delete all cards?',
      actions: [{
        label: 'Clear',
        destructive: true,
        onPress: () => {
          clearCollection();
          qc.invalidateQueries({ queryKey: ['collection'] });
          qc.invalidateQueries({ queryKey: ['collection-value'] });
        },
      }],
    });

  return (
    <View style={[styles.container, { backgroundColor: theme.bg }]}>
      <View style={[styles.header, { backgroundColor: theme.surface }]}>
        <View>
          <Text style={[styles.count, { color: theme.textSecondary }]}>{entries.length} cards</Text>
          <Text style={[styles.value, { color: theme.accent }]}>${totalValue.toFixed(2)}</Text>
        </View>
        <View style={styles.headerBtns}>
          <TouchableOpacity style={[styles.headerBtn, { backgroundColor: theme.surfaceAlt }]} onPress={handleImport}>
            <Text style={[styles.headerBtnText, { color: theme.text }]}>Import</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.headerBtn, { backgroundColor: theme.surfaceAlt }]} onPress={handleExport}>
            <Text style={[styles.headerBtnText, { color: theme.text }]}>Export</Text>
          </TouchableOpacity>
          {__DEV__ && (
            <TouchableOpacity style={[styles.headerBtn, styles.headerBtnDanger]} onPress={handleClear}>
              <Text style={[styles.headerBtnText, { color: theme.text }]}>DEV: Clear</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
      <View style={styles.toolbar}>
        <View style={[styles.searchWrap, { backgroundColor: theme.surface }]}>
          <TextInput
            style={[styles.searchBar, { color: theme.text }]}
            placeholder="Search..."
            placeholderTextColor={theme.textSecondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
            clearButtonMode={Platform.OS === 'ios' ? 'while-editing' : 'never'}
            autoCorrect={false}
            keyboardAppearance={keyboardAppearance}
          />
          {Platform.OS !== 'ios' && searchQuery.length > 0 && (
            <TouchableOpacity
              onPress={() => setSearchQuery('')}
              style={styles.clearBtn}
              hitSlop={HIT_SLOP_8}
              accessibilityRole="button"
              accessibilityLabel="Clear search"
            >
              <Icon name="close" size={16} color={theme.textSecondary} />
            </TouchableOpacity>
          )}
        </View>
        <ColorFilter active={colorFilter} onChange={setColorFilter} />
      </View>
      <CardGrid entries={entries} onAddPress={handleAddPress} />

      <Modal visible={importProgress !== null} transparent animationType="fade">
        <View style={styles.overlay}>
          <View style={[styles.importCard, { backgroundColor: theme.surface }]}>
            <ActivityIndicator size="large" color={theme.accent} />
            <Text style={[styles.importTitle, { color: theme.text }]}>Importing Cards</Text>
            <Text style={[styles.importCount, { color: theme.accent }]}>
              {importProgress?.current ?? 0} / {importProgress?.total ?? 0}
            </Text>
            <Text style={[styles.importName, { color: theme.textSecondary }]} numberOfLines={1}>
              {importProgress?.currentName}
            </Text>
            <View style={[styles.progressTrack, { backgroundColor: theme.surfaceAlt }]}>
              <View style={[styles.progressFill, { width: `${progress * 100}%`, backgroundColor: theme.accent }]} />
            </View>
          </View>
        </View>
      </Modal>
      {sheet.node}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    minHeight: MIN_TOUCH + spacing.lg,
  },
  count: { fontSize: 13 },
  value: { fontSize: 16, fontWeight: '700' },
  headerBtns: { flexDirection: 'row', gap: spacing.sm },
  headerBtn: {
    borderRadius: radius.sm + 2,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    minHeight: MIN_TOUCH,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerBtnText: { fontSize: font.small, fontWeight: '600' },
  headerBtnDanger: { backgroundColor: '#7a1a1a' },
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  searchWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    minHeight: MIN_TOUCH,
  },
  searchBar: { flex: 1, paddingVertical: spacing.sm, fontSize: font.body },
  clearBtn: { paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  clearBtnText: { fontSize: font.body, fontWeight: '600' },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', alignItems: 'center', justifyContent: 'center' },
  importCard: {
    borderRadius: radius.xl,
    padding: spacing.xl + 4,
    width: 280,
    alignItems: 'center',
    gap: spacing.md,
  },
  importTitle: { fontSize: 17, fontWeight: '700' },
  importCount: { fontSize: font.hero, fontWeight: '700' },
  importName: { fontSize: 13, maxWidth: 220, textAlign: 'center' },
  progressTrack: { width: '100%', height: 4, borderRadius: 2, overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 2 },
});
