import {
  View,
  TextInput,
  FlatList,
  Text,
  StyleSheet,
  Platform,
  useWindowDimensions,
  type ListRenderItem,
} from 'react-native';
import { Skeleton } from '../../src/components/Skeleton';
import { useCallback, useMemo, useState } from 'react';
import { CardRow } from '../../src/components/CardRow';
import { CardTile } from '../../src/components/CardTile';
import { useScryfallSearch } from '../../src/api/hooks';
import { useKeyboardAppearance, useTheme } from '../../src/theme/useTheme';
import { useStore } from '../../src/store/useStore';
import { useDebouncedValue } from '../../src/hooks/useDebouncedValue';
import type { CachedCard } from '../../src/db/cards';

// row 64 minHeight + 10*2 padding + 8 marginBottom — keep in sync with CardRow.
const ROW_HEIGHT = 84;
const LIST_PADDING = 8;

const keyExtractor = (c: CachedCard) => c.scryfall_id;
const getListItemLayout = (_: ArrayLike<CachedCard> | null | undefined, index: number) =>
  ({ length: ROW_HEIGHT, offset: ROW_HEIGHT * index, index });
const renderListItem: ListRenderItem<CachedCard> = ({ item }) => <CardRow card={item} />;

export default function SearchScreen() {
  const theme = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const viewMode = useStore((s) => s.searchViewMode);
  const gridCols = useStore((s) => s.searchGridCols);
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebouncedValue(query, 350);
  const { data: results = [], isLoading } = useScryfallSearch(debouncedQuery);
  const screenWidth = useWindowDimensions().width;

  const tileStyle = useMemo(
    () => ({ width: (screenWidth - LIST_PADDING * 2) / gridCols }),
    [screenWidth, gridCols],
  );

  const renderGridItem: ListRenderItem<CachedCard> = useCallback(
    ({ item }) => <CardTile card={item} style={tileStyle} />,
    [tileStyle],
  );

  const inputStyle = useMemo(
    () => [styles.input, { backgroundColor: theme.surface, color: theme.text }],
    [theme.surface, theme.text],
  );

  const ListEmpty = useCallback(() => {
    if (isLoading) return null;
    const q = query.trim();
    const msg = q.length === 0 ? 'Type to search Scryfall.' : q.length === 1 ? 'Keep typing…' : `No results for “${q}”.`;
    return <Text style={[styles.empty, { color: theme.textSecondary }]}>{msg}</Text>;
  }, [isLoading, query, theme.textSecondary]);

  const isGrid = viewMode === 'grid';

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      <View style={styles.inputRow}>
        <TextInput
          style={inputStyle}
          value={query}
          onChangeText={setQuery}
          placeholder='Search cards (try a scryfall query!)'
          placeholderTextColor={theme.textSecondary}
          autoCorrect={false}
          keyboardAppearance={keyboardAppearance}
        />
      </View>

      {isLoading && (
        // Skeleton placeholders matching the active view's row/tile size.
        // Reserves the eventual content height so results don't push the
        // first row down on arrival.
        isGrid ? (
          <View style={[styles.list, styles.skelGrid]}>
            {Array.from({ length: gridCols * 2 }).map((_, i) => (
              <Skeleton
                key={i}
                width={(screenWidth - LIST_PADDING * 2) / gridCols - 4}
                height={((screenWidth - LIST_PADDING * 2) / gridCols - 4) * 1.4}
                radius={6}
                style={{ margin: 2 }}
              />
            ))}
          </View>
        ) : (
          <View style={styles.list}>
            {Array.from({ length: 6 }).map((_, i) => (
              <View key={i} style={styles.skelRow}>
                <Skeleton width={48} height={68} radius={4} />
                <View style={{ flex: 1, marginLeft: 12, gap: 6 }}>
                  <Skeleton height={14} width="65%" />
                  <Skeleton height={11} width="40%" />
                </View>
              </View>
            ))}
          </View>
        )
      )}

      <FlatList
        key={isGrid ? `grid-${gridCols}` : 'list'}
        data={results}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.list}
        renderItem={isGrid ? renderGridItem : renderListItem}
        numColumns={isGrid ? gridCols : 1}
        columnWrapperStyle={isGrid && gridCols > 1 ? styles.rowLeft : undefined}
        getItemLayout={isGrid ? undefined : getListItemLayout}
        initialNumToRender={isGrid ? gridCols * 4 : 10}
        maxToRenderPerBatch={isGrid ? gridCols * 4 : 10}
        windowSize={7}
        removeClippedSubviews={Platform.OS === 'android'}
        ListEmptyComponent={ListEmpty}
        keyboardShouldPersistTaps="handled"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  inputRow: { padding: 12 },
  input: { borderRadius: 8, padding: 12, fontSize: 14 },
  list: { padding: 8 },
  skelGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  skelRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8 },
  rowLeft: { justifyContent: 'flex-start' },
  empty: { textAlign: 'center', marginTop: 40 },
});
