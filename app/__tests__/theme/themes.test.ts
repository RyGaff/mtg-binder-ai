jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
}));

import { themes } from '../../src/theme/themes';

describe('themes', () => {
  it('dark theme has expected bg color', () => {
    expect(themes.dark.bg).toBe('#111318');
  });

  it('light theme has expected bg color', () => {
    expect(themes.light.bg).toBe('#f2f2f7');
  });

  it('amoled theme has expected bg color', () => {
    expect(themes.amoled.bg).toBe('#000000');
  });

  it('all themes have the same accent color', () => {
    expect(themes.dark.accent).toBe('#4ecdc4');
    expect(themes.light.accent).toBe('#4ecdc4');
    expect(themes.amoled.accent).toBe('#4ecdc4');
  });

  it('all themes have all required keys', () => {
    const keys = ['bg', 'surface', 'surfaceAlt', 'border', 'text', 'textSecondary', 'accent'];
    for (const theme of Object.values(themes)) {
      for (const key of keys) {
        expect(theme).toHaveProperty(key);
      }
    }
  });
});

import { useStore } from '../../src/store/useStore';

describe('useStore theme', () => {
  afterEach(() => {
    useStore.getState().setTheme('dark');
  });

  it('default theme is dark', () => {
    expect(useStore.getState().theme).toBe('dark');
  });

  it('setTheme updates theme', () => {
    useStore.getState().setTheme('light');
    expect(useStore.getState().theme).toBe('light');
  });
});

import { swatches } from '../../src/theme/swatches';

describe('swatches', () => {
  it('each palette has exactly 24 colors', () => {
    for (const palette of Object.values(swatches)) {
      expect(palette).toHaveLength(24);
    }
  });

  it('all palette colors are valid hex strings', () => {
    for (const palette of Object.values(swatches)) {
      for (const color of palette) {
        expect(color).toMatch(/^#[0-9a-fA-F]{6}$/);
      }
    }
  });

  it('has the required palette keys', () => {
    expect(swatches).toHaveProperty('background');
    expect(swatches).toHaveProperty('surface');
    expect(swatches).toHaveProperty('border');
    expect(swatches).toHaveProperty('text');
    expect(swatches).toHaveProperty('textSecondary');
    expect(swatches).toHaveProperty('accent');
  });
});

import type { CustomTheme } from '../../src/theme/themes';

describe('CustomTheme type', () => {
  it('CustomTheme has all required keys', () => {
    const custom: CustomTheme = {
      name: 'custom-0',
      label: 'My Theme',
      bg: '#111318',
      surface: '#1a1c23',
      surfaceAlt: '#252830',
      border: '#2a2d38',
      text: '#ffffff',
      textSecondary: '#888888',
      accent: '#4ecdc4',
    };
    expect(custom.name).toBe('custom-0');
    expect(custom.label).toBe('My Theme');
    expect(custom.bg).toBe('#111318');
  });
});
