import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ThemeName, CustomTheme } from '../theme/themes';
import type { CachedCard } from '../db/cards';

type ColorFilter = 'W' | 'U' | 'B' | 'R' | 'G' | 'C' | 'all';
type SortOption = 'name' | 'value' | 'set' | 'added';
export type EmbeddingStatus = 'idle' | 'downloading' | 'error';
export type SearchViewMode = 'list' | 'grid';
export type SearchGridCols = 1 | 2 | 3 | 4 | 5;
export type DeckListMode = 'banner' | 'compact';
export type DeckViewMode = 'list' | 'grid';
type ThemeSlots = [CustomTheme | null, CustomTheme | null, CustomTheme | null];
type NavDir = 'initial' | 'forward' | 'backward';
type TrailEntry = { id: string; name: string };

type Store = {
  // Binder
  colorFilter: ColorFilter;
  setColorFilter: (color: ColorFilter) => void;
  sortOption: SortOption;
  setSortOption: (sort: SortOption) => void;

  // Deck builder
  activeDeckId: number | null;
  setActiveDeckId: (id: number | null) => void;

  // Scanner
  lastScannedId: string | null;
  setLastScannedId: (id: string | null) => void;

  // Recent scans (in-memory)
  recentScans: CachedCard[];
  addRecentScan: (card: CachedCard) => void;

  // Embeddings
  embeddingStatus: EmbeddingStatus;
  setEmbeddingStatus: (status: EmbeddingStatus) => void;

  // Search view preferences
  searchViewMode: SearchViewMode;
  setSearchViewMode: (mode: SearchViewMode) => void;
  searchGridCols: SearchGridCols;
  setSearchGridCols: (cols: SearchGridCols) => void;

  // Deck list view preference
  deckListMode: DeckListMode;
  setDeckListMode: (mode: DeckListMode) => void;

  // Per-deck view mode (list vs. grid). Keyed by deckId; missing entries fall
  // back to 'list'. Persisted so the user's per-deck choice survives restarts.
  deckViewModes: Record<number, DeckViewMode>;
  setDeckViewMode: (deckId: number, mode: DeckViewMode) => void;

  // Card detail breadcrumb trail (in-memory)
  cardTrail: TrailEntry[];
  pushCardTrail: (entry: TrailEntry) => void;
  clearCardTrail: () => void;
  /** Set by in-modal navigators (synergy, similar, printings) before router.replace
      so the modal's beforeRemove listener doesn't treat the replace as a dismiss. */
  suppressTrailReset: boolean;
  markInternalTrailNav: () => void;
  /** Direction of the last in-modal navigation, consumed by the card screen on mount
      to animate a horizontal slide. Reset to 'initial' after each animation plays. */
  lastCardNavDir: NavDir;
  setLastCardNavDir: (d: NavDir) => void;

  // Theme
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
  customThemes: ThemeSlots;
  setCustomTheme: (index: 0 | 1 | 2, theme: CustomTheme) => void;
  deleteCustomTheme: (index: 0 | 1 | 2) => void;
};

export const useStore = create<Store>()(
  persist(
    (set) => ({
      colorFilter: 'all',
      setColorFilter: (colorFilter) => set({ colorFilter }),
      sortOption: 'name',
      setSortOption: (sortOption) => set({ sortOption }),
      activeDeckId: null,
      setActiveDeckId: (activeDeckId) => set({ activeDeckId }),
      lastScannedId: null,
      setLastScannedId: (lastScannedId) => set({ lastScannedId }),
      recentScans: [],
      addRecentScan: (card) =>
        set((state) => ({
          recentScans: [card, ...state.recentScans.filter((c) => c.scryfall_id !== card.scryfall_id)].slice(0, 10),
        })),
      embeddingStatus: 'idle',
      setEmbeddingStatus: (embeddingStatus) => set({ embeddingStatus }),
      cardTrail: [],
      pushCardTrail: (entry) =>
        set((state) => {
          const last = state.cardTrail[state.cardTrail.length - 1];
          if (last && last.id === entry.id) return state;
          const existing = state.cardTrail.findIndex((t) => t.id === entry.id);
          if (existing >= 0) return { cardTrail: state.cardTrail.slice(0, existing + 1) };
          return { cardTrail: [...state.cardTrail, entry].slice(-5) };
        }),
      clearCardTrail: () => set({ cardTrail: [], lastCardNavDir: 'initial' }),
      suppressTrailReset: false,
      markInternalTrailNav: () => set({ suppressTrailReset: true, lastCardNavDir: 'forward' }),
      lastCardNavDir: 'initial',
      setLastCardNavDir: (lastCardNavDir) => set({ lastCardNavDir }),
      searchViewMode: 'list',
      setSearchViewMode: (searchViewMode) => set({ searchViewMode }),
      searchGridCols: 3,
      setSearchGridCols: (searchGridCols) => set({ searchGridCols }),
      deckListMode: 'banner',
      setDeckListMode: (deckListMode) => set({ deckListMode }),
      deckViewModes: {},
      setDeckViewMode: (deckId, mode) =>
        set((state) => ({ deckViewModes: { ...state.deckViewModes, [deckId]: mode } })),
      theme: 'dark',
      setTheme: (theme) => set({ theme }),
      customThemes: [null, null, null],
      setCustomTheme: (index, theme) =>
        set((state) => {
          const next = [...state.customThemes] as ThemeSlots;
          next[index] = theme;
          return { customThemes: next };
        }),
      deleteCustomTheme: (index) =>
        set((state) => {
          const next = [...state.customThemes] as ThemeSlots;
          next[index] = null;
          return { customThemes: next };
        }),
    }),
    {
      name: 'app-store',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        theme: state.theme,
        customThemes: state.customThemes,
        searchViewMode: state.searchViewMode,
        searchGridCols: state.searchGridCols,
        deckListMode: state.deckListMode,
        deckViewModes: state.deckViewModes,
      }),
      merge: (persistedState, currentState) => ({
        ...currentState,
        ...(persistedState as Partial<Store>),
      }),
    }
  )
);
