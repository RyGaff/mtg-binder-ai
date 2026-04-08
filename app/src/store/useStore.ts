import { create } from 'zustand';

type ColorFilter = 'W' | 'U' | 'B' | 'R' | 'G' | 'C' | 'all';
type SortOption = 'name' | 'value' | 'set' | 'added';

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
};

export const useStore = create<Store>((set) => ({
  colorFilter: 'all',
  setColorFilter: (colorFilter) => set({ colorFilter }),
  sortOption: 'name',
  setSortOption: (sortOption) => set({ sortOption }),
  activeDeckId: null,
  setActiveDeckId: (activeDeckId) => set({ activeDeckId }),
  lastScannedId: null,
  setLastScannedId: (lastScannedId) => set({ lastScannedId }),
}));
