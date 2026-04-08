import {
  View,
  Text,
  TextInput,
  Modal,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useState } from 'react';
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
  addToCollection,
  clearCollection,
} from '../../src/db/collection';
import { upsertCard, getCardBySetNumber, getCardById } from '../../src/db/cards';
import {
  serializeToJson,
  serializeToCsv,
  parseImportFile,
} from '../../src/export/collection';
import { fetchCardByName, fetchCardBySetNumber } from '../../src/api/scryfall';

export default function BinderScreen() {
  const router = useRouter();
  const qc = useQueryClient();
  const { colorFilter, setColorFilter } = useStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [importProgress, setImportProgress] = useState<{
    current: number;
    total: number;
    currentName: string;
  } | null>(null);

  const { data: entries = [] } = useQuery({
    queryKey: ['collection', colorFilter, searchQuery],
    queryFn: () => {
      if (searchQuery.trim()) return searchCollection(searchQuery.trim());
      return colorFilter === 'all' ? getCollection() : getCollectionByColor(colorFilter);
    },
  });

  const { data: totalValue = 0 } = useQuery({
    queryKey: ['collection-value'],
    queryFn: getCollectionTotalValue,
  });

  const exportAs = async (format: 'json' | 'csv') => {
    try {
      const content =
        format === 'json' ? serializeToJson(entries) : serializeToCsv(entries);
      const path = `${FileSystem.cacheDirectory}collection.${format}`;
      await FileSystem.writeAsStringAsync(path, content, {
        encoding: FileSystem.EncodingType.UTF8,
      });
      await Sharing.shareAsync(path);
    } catch {
      Alert.alert('Export Failed', 'Could not export collection.');
    }
  };

  const handleExport = () => {
    Alert.alert('Export Format', 'Choose a format', [
      { text: 'JSON', onPress: () => exportAs('json') },
      { text: 'CSV', onPress: () => exportAs('csv') },
      { text: 'Cancel', style: 'cancel' },
    ]);
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

    let added = 0;
    let failed = 0;
    setImportProgress({ current: 0, total: rows.length, currentName: '' });

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      setImportProgress({ current: i + 1, total: rows.length, currentName: row.name });
      try {
        let scryfallId = row.scryfall_id;
        if (!scryfallId) {
          // Check local cache first to avoid unnecessary Scryfall requests
          const cached =
            row.set_code && row.collector_number
              ? getCardBySetNumber(row.set_code, row.collector_number)
              : null;
          if (cached) {
            scryfallId = cached.scryfall_id;
          } else {
            // Throttle: 100ms between network calls to stay under Scryfall rate limit
            await new Promise((r) => setTimeout(r, 100));
            const card =
              row.set_code && row.collector_number
                ? await fetchCardBySetNumber(row.set_code, row.collector_number)
                : await fetchCardByName(row.name);
            upsertCard(card);
            scryfallId = card.scryfall_id;
          }
        } else if (!getCardById(scryfallId)) {
          // scryfall_id present but card not cached yet (e.g. our own JSON export)
          await new Promise((r) => setTimeout(r, 100));
          const card = await fetchCardBySetNumber(
            row.set_code ?? '',
            row.collector_number ?? ''
          ).catch(() => fetchCardByName(row.name));
          upsertCard(card);
        }
        addToCollection({
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

    setImportProgress(null);
    qc.invalidateQueries({ queryKey: ['collection'] });
    qc.invalidateQueries({ queryKey: ['collection-value'] });
    const msg = failed > 0
      ? `Added ${added} cards. ${failed} could not be found.`
      : `Added ${added} cards.`;
    Alert.alert('Import Complete', msg);
  };

  const progress = importProgress
    ? importProgress.current / importProgress.total
    : 0;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.count}>{entries.length} cards</Text>
          <Text style={styles.value}>${totalValue.toFixed(2)}</Text>
        </View>
        <View style={styles.headerBtns}>
          <TouchableOpacity style={styles.headerBtn} onPress={handleImport}>
            <Text style={styles.headerBtnText}>Import</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerBtn} onPress={handleExport}>
            <Text style={styles.headerBtnText}>Export</Text>
          </TouchableOpacity>
          {__DEV__ && (
            <TouchableOpacity
              style={[styles.headerBtn, styles.headerBtnDanger]}
              onPress={() =>
                Alert.alert('Clear Collection', 'Delete all cards?', [
                  { text: 'Cancel', style: 'cancel' },
                  {
                    text: 'Clear',
                    style: 'destructive',
                    onPress: () => {
                      clearCollection();
                      qc.invalidateQueries({ queryKey: ['collection'] });
                      qc.invalidateQueries({ queryKey: ['collection-value'] });
                    },
                  },
                ])
              }
            >
              <Text style={styles.headerBtnText}>DEV: Clear</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
      <View style={styles.toolbar}>
        <TextInput
          style={styles.searchBar}
          placeholder="Search..."
          placeholderTextColor="#555"
          value={searchQuery}
          onChangeText={setSearchQuery}
          clearButtonMode="while-editing"
          autoCorrect={false}
        />
        <ColorFilter active={colorFilter} onChange={setColorFilter} />
      </View>
      <CardGrid entries={entries} onAddPress={() => router.push('/search')} />

      <Modal visible={importProgress !== null} transparent animationType="fade">
        <View style={styles.overlay}>
          <View style={styles.importCard}>
            <ActivityIndicator size="large" color="#4ecdc4" />
            <Text style={styles.importTitle}>Importing Cards</Text>
            <Text style={styles.importCount}>
              {importProgress?.current ?? 0} / {importProgress?.total ?? 0}
            </Text>
            <Text style={styles.importName} numberOfLines={1}>
              {importProgress?.currentName}
            </Text>
            <View style={styles.progressTrack}>
              <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111318' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#1a1c23',
  },
  count: { color: '#aaa', fontSize: 13 },
  value: { color: '#4ecdc4', fontSize: 16, fontWeight: '700' },
  headerBtns: { flexDirection: 'row', gap: 8 },
  headerBtn: {
    backgroundColor: '#252830',
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  headerBtnText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  headerBtnDanger: { backgroundColor: '#7a1a1a' },
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    gap: 8,
  },
  searchBar: {
    flex: 1,
    backgroundColor: '#1a1c23',
    color: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    fontSize: 14,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  importCard: {
    backgroundColor: '#1a1c23',
    borderRadius: 16,
    padding: 28,
    width: 280,
    alignItems: 'center',
    gap: 12,
  },
  importTitle: { color: '#fff', fontSize: 17, fontWeight: '700' },
  importCount: { color: '#4ecdc4', fontSize: 22, fontWeight: '700' },
  importName: { color: '#aaa', fontSize: 13, maxWidth: 220, textAlign: 'center' },
  progressTrack: {
    width: '100%',
    height: 4,
    backgroundColor: '#252830',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4ecdc4',
    borderRadius: 2,
  },
});
