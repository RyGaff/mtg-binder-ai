import { useStore } from '../store/useStore';
import { themes, type Theme, type ThemeName } from './themes';

export function useTheme(): Theme {
  const theme = useStore((s) => s.theme);
  const builtInTheme = theme as Exclude<ThemeName, `custom-${number}`>;
  return themes[builtInTheme] ?? themes.dark;
}