import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Modal, Platform,
  Pressable, ScrollView, StyleSheet, Text, TextInput,
  TouchableOpacity, View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useQueryClient } from '@tanstack/react-query';
import { upsertCard, type CachedCard } from '../db/cards';
import { addCardToDeck, createDeck, type Board } from '../db/decks';
import { parseDeckText, resolveDeckCards, type ParsedLine } from '../utils/deckImport';
import { fetchDeckFromUrl, parseDeckSourceUrl } from '../api/deckSources';
import { LruCache } from '../api/lruCache';
import { useTheme } from '../theme/useTheme';

// Duplicated from app/(tabs)/decks.tsx — small const list, importing across the
// tab/component boundary is more friction than its worth.
const FORMATS = ['Commander', 'Standard', 'Modern', 'Legacy', 'Vintage', 'Pioneer', 'Pauper', 'Draft', 'Other'] as const;
type Format = typeof FORMATS[number];

// Mirror of BOARD_HEADERS in deckImport.ts. Used only for the marker layer to
// recognize header lines so we can color them gold. Kept local to avoid
// exporting an internal from deckImport.ts.
const BOARD_HEADERS = new Set<string>([
  'commander', 'commanders', 'deck', 'main', 'mainboard', 'main deck',
  'sideboard', 'side', 'considering', 'maybeboard', 'maybe',
]);

const LINE_RE = /^(\d+)x?\s+(.+?)(?:\s+\([A-Z0-9]+\)\s+\S+)?\s*$/;

type LineStatus = 'header' | 'valid' | 'pending' | 'unresolved' | 'malformed' | 'comment' | 'blank';

type RawLine = {
  raw: string;       // text as displayed (preserves user's original whitespace per line)
  trimmed: string;
  isHeader: boolean;
  isComment: boolean;
  isBlank: boolean;
  isMalformed: boolean;
  parsedName: string | null;     // present when line parsed as `<qty> <name>`
  parsedQty: number | null;
  parsedBoard: Board;            // board active at this point in the text
};

type Mode = 'paste' | 'url';

type Props = {
  visible: boolean;
  onClose: () => void;
  /** Called with the new deck id once import is complete (so the caller can navigate). */
  onImported?: (deckId: number) => void;
};

const FONT_FAMILY = Platform.select({ ios: 'Menlo', android: 'monospace' });
const FONT_SIZE = 14;
const LINE_HEIGHT = 20;
const TEXTAREA_PADDING = 12;

// Print fetched ParsedLine[] back into pastable decklist text so the marker
// pipeline can take over after a URL fetch. Boards print in a stable order.
const BOARD_ORDER: Board[] = ['commander', 'main', 'side', 'considering'];
const BOARD_LABEL: Record<Board, string> = {
  commander: 'Commander',
  main: 'Deck',
  side: 'Sideboard',
  considering: 'Considering',
};
function synthesizeDecklistText(lines: ParsedLine[]): string {
  const grouped: Record<Board, ParsedLine[]> = { commander: [], main: [], side: [], considering: [] };
  for (const l of lines) grouped[l.board].push(l);
  const out: string[] = [];
  for (const b of BOARD_ORDER) {
    const list = grouped[b];
    if (!list.length) continue;
    if (out.length) out.push('');
    out.push(BOARD_LABEL[b]);
    for (const l of list) out.push(`${l.quantity} ${l.name}`);
  }
  return out.join('\n');
}

// Walk the raw text once to derive per-line render info AND the board context
// for each parsed line. Mirrors parseDeckText's loop so statuses match exactly.
function annotateLines(text: string): RawLine[] {
  const out: RawLine[] = [];
  let board: Board = 'main';
  for (const raw of text.split(/\r?\n/)) {
    const trimmed = raw.trim();
    if (!trimmed) {
      out.push({ raw, trimmed, isHeader: false, isComment: false, isBlank: true, isMalformed: false, parsedName: null, parsedQty: null, parsedBoard: board });
      continue;
    }
    if (trimmed.startsWith('//') || trimmed.startsWith('#')) {
      out.push({ raw, trimmed, isHeader: false, isComment: true, isBlank: false, isMalformed: false, parsedName: null, parsedQty: null, parsedBoard: board });
      continue;
    }
    const headerKey = trimmed.toLowerCase().replace(/[:\-]+$/, '').trim();
    if (BOARD_HEADERS.has(headerKey)) {
      const next = headerKey;
      // Apply the same mapping as deckImport for board context to subsequent lines.
      if (next === 'commander' || next === 'commanders') board = 'commander';
      else if (next === 'deck' || next === 'main' || next === 'mainboard' || next === 'main deck') board = 'main';
      else if (next === 'sideboard' || next === 'side') board = 'side';
      else if (next === 'considering' || next === 'maybeboard' || next === 'maybe') board = 'considering';
      out.push({ raw, trimmed, isHeader: true, isComment: false, isBlank: false, isMalformed: false, parsedName: null, parsedQty: null, parsedBoard: board });
      continue;
    }
    const m = trimmed.match(LINE_RE);
    if (!m) {
      out.push({ raw, trimmed, isHeader: false, isComment: false, isBlank: false, isMalformed: true, parsedName: null, parsedQty: null, parsedBoard: board });
      continue;
    }
    const qty = parseInt(m[1], 10);
    const name = m[2].trim();
    if (!(qty > 0 && name)) {
      out.push({ raw, trimmed, isHeader: false, isComment: false, isBlank: false, isMalformed: true, parsedName: null, parsedQty: null, parsedBoard: board });
      continue;
    }
    out.push({ raw, trimmed, isHeader: false, isComment: false, isBlank: false, isMalformed: false, parsedName: name, parsedQty: qty, parsedBoard: board });
  }
  return out;
}

export function ImportDeckSheet({ visible, onClose, onImported }: Props): ReactElement {
  const t = useTheme();
  const qc = useQueryClient();
  const insets = useSafeAreaInsets();

  const [mode, setMode] = useState<Mode>('paste');
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [format, setFormat] = useState<Format>('Commander');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [urlError, setUrlError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importDone, setImportDone] = useState(0);
  const [importError, setImportError] = useState<string | null>(null);
  // Bumped whenever the resolver completes — drives marker re-render via cache.
  const [resolveTick, setResolveTick] = useState(0);

  // Resolved-card cache (lowercased name → CachedCard | null). Survives the
  // life of the component instance; null = known-bad so we don't re-request.
  // Bounded so a long paste session can't grow without bound — 2000 unique
  // names covers any realistic decklist plus retries.
  const cacheRef = useRef<LruCache<string, CachedCard | null>>(new LruCache(2000));
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fetchAbortRef = useRef<AbortController | null>(null);

  const textInputRef = useRef<TextInput | null>(null);
  const urlInputRef = useRef<TextInput | null>(null);

  // Auto-focus on open. 30ms delay lets Modal mount first.
  useEffect(() => {
    if (!visible) return;
    const id = setTimeout(() => {
      if (mode === 'paste') textInputRef.current?.focus();
      else urlInputRef.current?.focus();
    }, 30);
    return () => clearTimeout(id);
  }, [visible, mode]);

  // Cancel pending work when the component fully unmounts.
  useEffect(() => () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    abortRef.current?.abort();
    fetchAbortRef.current?.abort();
  }, []);

  const annotated = useMemo(() => annotateLines(text), [text]);

  // Total height of the marker/textarea content so the ScrollView wrapping them
  // knows when to enable scrolling. Floor = ~7 lines so an empty paste area still
  // has a sensible height.
  const textareaContentMinHeight = useMemo(
    () => Math.max(annotated.length, 7) * LINE_HEIGHT + TEXTAREA_PADDING * 2,
    [annotated.length],
  );

  // Derive per-line status using the cache (resolveTick forces recompute).
  const statuses = useMemo<LineStatus[]>(() => {
    const cache = cacheRef.current;
    return annotated.map((l) => {
      if (l.isBlank) return 'blank';
      if (l.isComment) return 'comment';
      if (l.isHeader) return 'header';
      if (l.isMalformed) return 'malformed';
      if (l.parsedName) {
        const lower = l.parsedName.toLowerCase();
        if (cache.has(lower)) return cache.get(lower) ? 'valid' : 'unresolved';
        return 'pending';
      }
      return 'malformed';
    });
    // resolveTick is intentionally a dependency — it invalidates this memo
    // when the cache changes, since the cache itself is a ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [annotated, resolveTick]);

  const parsedLineEntries = useMemo(
    () => annotated.filter((l) => l.parsedName !== null) as (RawLine & { parsedName: string; parsedQty: number })[],
    [annotated],
  );

  const resolvedCount = useMemo(() => {
    const cache = cacheRef.current;
    let c = 0;
    for (const l of parsedLineEntries) {
      if (cache.get(l.parsedName.toLowerCase())) c++;
    }
    return c;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parsedLineEntries, resolveTick]);

  const unresolvedCount = useMemo(() => {
    const cache = cacheRef.current;
    let c = 0;
    for (const l of parsedLineEntries) {
      const cached = cache.get(l.parsedName.toLowerCase());
      if (cached === null) c++;
    }
    return c;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parsedLineEntries, resolveTick]);

  const totalParsed = parsedLineEntries.length;

  // Debounced bulk resolver. Triggers on every text change; the most-recent
  // call wins because we abort the in-flight request on every keystroke.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    abortRef.current?.abort();

    const cache = cacheRef.current;
    const needed = new Set<string>();
    for (const l of parsedLineEntries) {
      const lower = l.parsedName.toLowerCase();
      if (!cache.has(lower)) needed.add(l.parsedName);
    }
    if (needed.size === 0) return;

    const controller = new AbortController();
    abortRef.current = controller;

    debounceRef.current = setTimeout(() => {
      (async () => {
        try {
          const result = await resolveDeckCards(Array.from(needed), controller.signal);
          if (controller.signal.aborted) return;
          for (const [lower, card] of result.resolved.entries()) {
            cache.set(lower, card);
          }
          for (const lower of result.unresolved) {
            // Only mark as null if not already resolved under that key (dedup safety).
            if (!cache.has(lower)) cache.set(lower, null);
          }
          setResolveTick((n) => n + 1);
        } catch (err) {
          if (controller.signal.aborted) return;
          // Mark all needed names as unresolved so the user sees feedback rather
          // than indefinite "pending" coloring on a network error.
          for (const original of needed) cache.set(original.toLowerCase(), null);
          setResolveTick((n) => n + 1);
        }
      })();
    }, 500);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [parsedLineEntries]);

  const close = () => onClose();

  const setNameFromUser = (v: string) => {
    setName(v);
    if (v.length > 0) setNameTouched(true);
    else setNameTouched(false);
  };

  const doFetchUrl = async (urlValue: string) => {
    const trimmed = urlValue.trim();
    if (!trimmed) return;
    if (!parseDeckSourceUrl(trimmed)) {
      setUrlError('Unsupported URL. Use a Moxfield or Archidekt deck URL.');
      return;
    }
    fetchAbortRef.current?.abort();
    const controller = new AbortController();
    fetchAbortRef.current = controller;
    setFetching(true);
    setUrlError(null);
    try {
      const fetched = await fetchDeckFromUrl(trimmed, controller.signal);
      if (controller.signal.aborted) return;
      if (!nameTouched && fetched.name) setName(fetched.name);
      const matched = (FORMATS as readonly string[]).includes(fetched.format)
        ? (fetched.format as Format)
        : 'Commander';
      setFormat(matched);
      setText(synthesizeDecklistText(fetched.lines));
      setMode('paste');
    } catch (err) {
      if (controller.signal.aborted) return;
      const msg = (err as Error)?.message ?? 'Fetch failed';
      setUrlError(msg);
    } finally {
      if (!controller.signal.aborted) setFetching(false);
    }
  };

  const onUrlBlur = () => {
    if (!url.trim()) return;
    if (parseDeckSourceUrl(url.trim())) void doFetchUrl(url);
  };

  const importDisabled = !name.trim() || resolvedCount === 0 || importing;

  const submit = async () => {
    if (importDisabled) return;
    setImporting(true);
    setImportDone(0);
    setImportError(null);
    try {
      const cache = cacheRef.current;
      const importable = parsedLineEntries
        .map((l) => ({ line: l, card: cache.get(l.parsedName.toLowerCase()) ?? null }))
        .filter((e): e is { line: typeof e.line; card: CachedCard } => e.card !== null);

      // Upsert all cards first so addCardToDeck's FK is satisfied.
      const seen = new Set<string>();
      for (const { card } of importable) {
        if (seen.has(card.scryfall_id)) continue;
        seen.add(card.scryfall_id);
        upsertCard(card);
      }

      const newId = createDeck({ name: name.trim(), format });

      for (let i = 0; i < importable.length; i++) {
        const { line, card } = importable[i];
        addCardToDeck({
          deck_id: newId,
          scryfall_id: card.scryfall_id,
          quantity: line.parsedQty,
          board: line.parsedBoard,
        });
        setImportDone(i + 1);
      }

      qc.invalidateQueries({ queryKey: ['decks'] });
      qc.invalidateQueries({ queryKey: ['deck-cards', newId] });

      setImporting(false);
      onImported?.(newId);
      onClose();
    } catch (err) {
      setImporting(false);
      setImportError((err as Error)?.message ?? 'Import failed');
    }
  };

  // Tap "N unknown" → jump caret to the first unresolved line.
  const jumpToFirstUnresolved = () => {
    const cache = cacheRef.current;
    let offset = 0;
    for (const l of annotated) {
      if (l.parsedName && cache.get(l.parsedName.toLowerCase()) === null) {
        textInputRef.current?.focus();
        // setNativeProps to set selection without changing the controlled value.
        textInputRef.current?.setNativeProps({ selection: { start: offset, end: offset + l.raw.length } });
        return;
      }
      offset += l.raw.length + 1; // +1 for the newline split
    }
  };

  const statusColor = (status: LineStatus): { color: string; fontStyle?: 'italic'; textDecorationLine?: 'underline'; opacity?: number } => {
    switch (status) {
      case 'header': return { color: '#d4a017' };
      case 'valid': return { color: t.success };
      case 'pending': return { color: t.textSecondary, fontStyle: 'italic' };
      case 'unresolved': return { color: t.danger, textDecorationLine: 'underline' };
      case 'malformed': return { color: t.danger };
      case 'comment': return { color: t.textSecondary, opacity: 0.6 };
      case 'blank':
      default: return { color: t.text };
    }
  };

  const segLabel = (m: Mode) => m === 'paste' ? 'Paste' : 'URL';

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      hardwareAccelerated
      statusBarTranslucent
      onRequestClose={close}
    >
      <View style={s.backdrop}>
        <Pressable style={StyleSheet.absoluteFill} onPress={close} />
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={s.kavWrap}
          pointerEvents="box-none"
        >
          <View
            style={[
              s.sheet,
              {
                backgroundColor: t.surface,
                borderColor: t.border,
                paddingTop: 6,
                paddingBottom: Math.max(insets.bottom, 12),
              },
            ]}
          >
            {/* Drag handle (decorative, also dismisses on tap). */}
            <TouchableOpacity activeOpacity={0.7} onPress={close} style={s.handleHit}>
              <View style={[s.handle, { backgroundColor: t.border }]} />
            </TouchableOpacity>

            <View style={s.titleRow}>
              <View style={s.titleSpacer} />
              <Text style={[s.title, { color: t.text }]}>Import deck</Text>
              <TouchableOpacity onPress={close} hitSlop={10} style={s.closeBtn}>
                <Text style={[s.closeText, { color: t.textSecondary }]}>✕</Text>
              </TouchableOpacity>
            </View>

            {/* Segmented Paste | URL */}
            <View style={[s.segWrap, { backgroundColor: t.surfaceAlt }]}>
              {(['paste', 'url'] as const).map((m) => {
                const active = mode === m;
                return (
                  <TouchableOpacity
                    key={m}
                    onPress={() => setMode(m)}
                    activeOpacity={0.8}
                    style={[
                      s.segBtn,
                      { backgroundColor: active ? t.accent : 'transparent' },
                    ]}
                  >
                    <Text style={[s.segText, { color: active ? '#fff' : t.text }]}>
                      {segLabel(m)}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>

            {/* Deck name */}
            <TextInput
              value={name}
              onChangeText={setNameFromUser}
              placeholder="Deck name"
              placeholderTextColor={t.textSecondary}
              autoCapitalize="words"
              editable={!importing}
              style={[s.input, { backgroundColor: t.bg, color: t.text, borderColor: t.border }]}
            />

            {/* Format chips */}
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={s.chipsRow}
            >
              {FORMATS.map((f) => {
                const active = f === format;
                return (
                  <TouchableOpacity
                    key={f}
                    onPress={() => setFormat(f)}
                    activeOpacity={0.8}
                    disabled={importing}
                    style={[
                      s.chip,
                      { backgroundColor: active ? t.accent : t.surfaceAlt },
                    ]}
                  >
                    <Text style={[s.chipText, { color: active ? '#fff' : t.text }]}>
                      {f}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            {/* Mode body */}
            {mode === 'paste' ? (
              <View style={[s.textareaWrap, { borderColor: t.border, backgroundColor: t.bg }]}>
                {/* Outer ScrollView so long decklists scroll past the keyboard. The
                    marker layer (absolute) and the TextInput (flow) both sit inside
                    contentContainer; their shared minHeight = lines × lineHeight so
                    both grow together. TextInput has scrollEnabled={false} so the
                    inner scroll doesn't fight the outer. */}
                <ScrollView
                  style={s.textareaScroll}
                  contentContainerStyle={[s.textareaContent, { minHeight: textareaContentMinHeight }]}
                  keyboardShouldPersistTaps="handled"
                  showsVerticalScrollIndicator
                >
                  <View pointerEvents="none" style={s.markerLayer}>
                    {annotated.length === 0 ? null : annotated.map((l, i) => (
                      <Text
                        key={i}
                        style={[s.markerLine, statusColor(statuses[i])]}
                      >
                        {l.raw.length === 0 ? ' ' : l.raw}
                      </Text>
                    ))}
                  </View>
                  <TextInput
                    ref={textInputRef}
                    value={text}
                    onChangeText={setText}
                    placeholder={'Paste from MTG Arena / MTGO / Moxfield…'}
                    placeholderTextColor={t.textSecondary}
                    multiline
                    scrollEnabled={false}
                    autoCorrect={false}
                    autoCapitalize="none"
                    spellCheck={false}
                    textAlignVertical="top"
                    selectionColor={t.accent}
                    editable={!importing}
                    style={[s.textareaInput, { color: 'transparent', minHeight: textareaContentMinHeight }]}
                  />
                </ScrollView>
              </View>
            ) : (
              <View style={s.urlBlock}>
                <View style={s.urlRow}>
                  <TextInput
                    ref={urlInputRef}
                    value={url}
                    onChangeText={(v) => { setUrl(v); if (urlError) setUrlError(null); }}
                    onBlur={onUrlBlur}
                    placeholder="https://moxfield.com/decks/… or https://archidekt.com/decks/…"
                    placeholderTextColor={t.textSecondary}
                    autoCorrect={false}
                    autoCapitalize="none"
                    spellCheck={false}
                    keyboardType="url"
                    editable={!fetching && !importing}
                    style={[s.input, s.urlInput, { backgroundColor: t.bg, color: t.text, borderColor: t.border }]}
                  />
                  <TouchableOpacity
                    onPress={() => void doFetchUrl(url)}
                    disabled={fetching || !url.trim() || importing}
                    activeOpacity={0.8}
                    style={[
                      s.fetchBtn,
                      { backgroundColor: t.accent },
                      (fetching || !url.trim() || importing) && { opacity: 0.5 },
                    ]}
                  >
                    {fetching ? (
                      <ActivityIndicator size="small" color="#fff" />
                    ) : (
                      <Text style={s.fetchBtnText}>Fetch</Text>
                    )}
                  </TouchableOpacity>
                </View>
                {urlError ? (
                  <Text style={[s.errorText, { color: t.danger }]}>{urlError}</Text>
                ) : null}
              </View>
            )}

            {/* Status line */}
            <View style={s.statusRow}>
              {importError ? (
                <Text style={[s.statusText, { color: t.danger }]} numberOfLines={2}>
                  {importError}
                </Text>
              ) : totalParsed > 0 ? (
                <Text style={[s.statusText, { color: t.textSecondary }]}>
                  {resolvedCount} of {totalParsed} found
                  {unresolvedCount > 0 ? (
                    <>
                      {' · '}
                      <Text
                        onPress={jumpToFirstUnresolved}
                        style={{ color: t.danger, textDecorationLine: 'underline' }}
                      >
                        {unresolvedCount} unknown
                      </Text>
                    </>
                  ) : null}
                </Text>
              ) : (
                <Text style={[s.statusText, { color: t.textSecondary }]}>
                  {mode === 'paste' ? 'Paste a decklist to begin.' : 'Paste a Moxfield or Archidekt URL.'}
                </Text>
              )}
            </View>

            {/* Primary action */}
            <TouchableOpacity
              onPress={submit}
              disabled={importDisabled}
              activeOpacity={0.85}
              style={[
                s.primaryBtn,
                { backgroundColor: t.accent },
                importDisabled && { opacity: 0.4 },
              ]}
            >
              {importing ? (
                <View style={s.primaryBtnInner}>
                  <ActivityIndicator size="small" color="#fff" />
                  <Text style={s.primaryBtnText}>
                    {`Importing ${importDone} / ${resolvedCount} …`}
                  </Text>
                </View>
              ) : (
                <Text style={s.primaryBtnText}>
                  {resolvedCount > 0 ? `Import ${resolvedCount} cards` : 'Import'}
                </Text>
              )}
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)' },
  kavWrap: { flex: 1, justifyContent: 'flex-end' },
  sheet: {
    flex: 1,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: 14,
    gap: 10,
  },
  handleHit: { alignItems: 'center', paddingVertical: 6 },
  handle: { width: 40, height: 4, borderRadius: 2 },
  titleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 4 },
  titleSpacer: { width: 28 },
  title: { fontSize: 17, fontWeight: '700', flex: 1, textAlign: 'center' },
  closeBtn: { width: 28, height: 28, alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 18, fontWeight: '600' },
  segWrap: { flexDirection: 'row', borderRadius: 10, padding: 3, gap: 3 },
  segBtn: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 8 },
  segText: { fontSize: 13, fontWeight: '700' },
  input: {
    height: 44,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: 12,
    fontSize: 15,
  },
  chipsRow: { gap: 6, paddingHorizontal: 2, alignItems: 'center' },
  chip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999 },
  chipText: { fontSize: 12, fontWeight: '600' },
  textareaWrap: {
    flex: 1,
    minHeight: 140,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    overflow: 'hidden',
  },
  // Outer ScrollView fills the wrap; contentContainer is `relative` so the
  // absolute markerLayer anchors to it (the TextInput renders in normal flow
  // beneath, with matching minHeight).
  textareaScroll: { flex: 1 },
  textareaContent: { position: 'relative' },
  markerLayer: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    padding: TEXTAREA_PADDING,
  },
  markerLine: {
    fontFamily: FONT_FAMILY,
    fontSize: FONT_SIZE,
    lineHeight: LINE_HEIGHT,
    includeFontPadding: false,
  },
  textareaInput: {
    flex: 1,
    fontFamily: FONT_FAMILY,
    fontSize: FONT_SIZE,
    lineHeight: LINE_HEIGHT,
    padding: TEXTAREA_PADDING,
    textAlignVertical: 'top',
    includeFontPadding: false,
  },
  urlBlock: { gap: 6 },
  urlRow: { flexDirection: 'row', gap: 8, alignItems: 'center' },
  urlInput: { flex: 1 },
  fetchBtn: { paddingHorizontal: 16, height: 44, borderRadius: 10, alignItems: 'center', justifyContent: 'center', minWidth: 72 },
  fetchBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  errorText: { fontSize: 12 },
  statusRow: { paddingHorizontal: 4, minHeight: 20 },
  statusText: { fontSize: 13 },
  primaryBtn: {
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 4,
  },
  primaryBtnInner: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  primaryBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});
