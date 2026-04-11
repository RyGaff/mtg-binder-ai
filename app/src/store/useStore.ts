import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ThemeName, CustomTheme } from '../theme/themes';

type ColorFilter = 'W' | 'U' | 'B' | 'R' | 'G' | 'C' | 'all';
type SortOption = 'name' | 'value' | 'set' | 'added';
export type EmbeddingStatus = 'idle' | 'downloading' | 'error';
type ThemeSlots = [CustomTheme | null, CustomTheme | null, CustomTheme | null];

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

  // Embeddings
  embeddingStatus: EmbeddingStatus;
  setEmbeddingStatus: (status: EmbeddingStatus) => void;

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
      embeddingStatus: 'idle',
      setEmbeddingStatus: (embeddingStatus) => set({ embeddingStatus }),
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
      partialize: (state) => ({ theme: state.theme, customThemes: state.customThemes }),
      merge: (persistedState, currentState) => ({
        ...currentState,
        ...(persistedState as Partial<Store>),
      }),
    }
  )
);
