import { useStore } from '../store/useStore';
import { themes, type Theme } from './themes';

export function useTheme(): Theme {
  return useStore((s) => {
    if (s.theme.startsWith('custom-')) {
      const index = Number(s.theme.split('-')[1]) as 0 | 1 | 2;
      return s.customThemes[index] ?? themes.dark;
    }
    return themes[s.theme as 'dark' | 'light' | 'amoled'] ?? themes.dark;
  });
}
