import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useStore } from '../src/store/useStore';
import { isDarkTheme, useKeyboardAppearance, useTheme } from '../src/theme/useTheme';
import { themes, type CustomTheme, type CustomThemeName } from '../src/theme/themes';
import { swatches } from '../src/theme/swatches';
import { Icon } from '../src/components/icons/Icon';

// Per-swatch contrast: pick the check color so it's readable on the swatch's bg.
function checkColorFor(hex: string): string {
  return isDarkTheme({ bg: hex } as never) ? '#ffffff' : '#000000';
}

type TokenKey = 'bg' | 'surface' | 'surfaceAlt' | 'border' | 'text' | 'textSecondary' | 'accent';

const TOKEN_ROWS: { key: TokenKey; label: string; palette: keyof typeof swatches }[] = [
  { key: 'bg', label: 'Background', palette: 'background' },
  { key: 'surface', label: 'Surface', palette: 'surface' },
  { key: 'surfaceAlt', label: 'Surface Alt', palette: 'surface' },
  { key: 'border', label: 'Border', palette: 'border' },
  { key: 'text', label: 'Text', palette: 'text' },
  { key: 'textSecondary', label: 'Secondary Text', palette: 'textSecondary' },
  { key: 'accent', label: 'Accent', palette: 'accent' },
];

type DraftColors = Record<TokenKey, string>;

export default function ThemeEditorScreen() {
  const router = useRouter();
  const t = useTheme();
  const keyboardAppearance = useKeyboardAppearance();
  const setCustomTheme = useStore((s) => s.setCustomTheme);
  const deleteCustomTheme = useStore((s) => s.deleteCustomTheme);
  const setTheme = useStore((s) => s.setTheme);
  const customThemes = useStore((s) => s.customThemes);
  const theme = useStore((s) => s.theme);
  const { slot: slotParam, mode } = useLocalSearchParams<{ slot: string; mode: 'new' | 'edit' }>();
  const slotNum = Number(slotParam);
  const isValidSlot = Number.isInteger(slotNum) && slotNum >= 0 && slotNum <= 2;
  const slot = (isValidSlot ? slotNum : 0) as 0 | 1 | 2;
  const customName = `custom-${slot}` as CustomThemeName;

  const existing = customThemes[slot];
  const base = existing ?? themes.dark;

  const [label, setLabel] = useState(existing?.label ?? '');
  const [colors, setColors] = useState<DraftColors>({
    bg: base.bg,
    surface: base.surface,
    surfaceAlt: base.surfaceAlt,
    border: base.border,
    text: base.text,
    textSecondary: base.textSecondary,
    accent: base.accent,
  });
  const [expandedToken, setExpandedToken] = useState<TokenKey | null>(null);

  useEffect(() => {
    if (!isValidSlot) {
      if (router.canGoBack()) router.back(); else router.replace('/');
    }
  }, [isValidSlot]);

  if (!isValidSlot) return null;

  const setColor = (key: TokenKey, value: string) => {
    setColors((prev) => ({ ...prev, [key]: value }));
  };

  const toggleExpanded = (key: TokenKey) => {
    setExpandedToken((prev) => (prev === key ? null : key));
  };

  const handleSave = () => {
    // Inherit shared (non-token) fields like foilAccent/danger/success from the dark base.
    const custom: CustomTheme = { ...themes.dark, name: customName, label, ...colors };
    setCustomTheme(slot, custom);
    if (mode === 'new' || theme === customName) {
      setTheme(customName);
    }
    if (router.canGoBack()) router.back(); else router.replace('/');
  };

  const handleDelete = () => {
    if (theme === customName) setTheme('dark');
    deleteCustomTheme(slot);
    if (router.canGoBack()) router.back(); else router.replace('/');
  };

  return (
    <View style={[styles.screen, { backgroundColor: t.bg }]}>
      {/* Close button */}
      <TouchableOpacity
        style={styles.closeBtn}
        onPress={() => (router.canGoBack() ? router.back() : router.replace('/'))}
        accessibilityLabel="Close without saving"
        accessibilityRole="button"
      >
        <Icon name="close" size={20} color={t.textSecondary} />
      </TouchableOpacity>

      <Text style={[styles.screenTitle, { color: t.text }]}>
        {mode === 'edit' ? 'Edit Theme' : 'New Theme'}
      </Text>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Live preview strip */}
        <View style={styles.preview}>
          {(['bg', 'surface', 'surfaceAlt', 'border', 'accent'] as TokenKey[]).map((key) => (
            <View key={key} style={[styles.previewSegment, { backgroundColor: colors[key] }]} />
          ))}
        </View>

        {/* Name field */}
        <TextInput
          style={[styles.nameInput, { backgroundColor: t.surface, color: t.text, borderColor: t.border }]}
          placeholder="Theme name"
          placeholderTextColor={t.textSecondary}
          value={label}
          onChangeText={setLabel}
          maxLength={20}
          keyboardAppearance={keyboardAppearance}
          returnKeyType="done"
        />

        {/* Token rows */}
        {TOKEN_ROWS.map(({ key, label: tokenLabel, palette }) => {
          const isOpen = expandedToken === key;
          const currentColor = colors[key];
          return (
            <View key={key}>
              <TouchableOpacity
                style={[styles.tokenRow, { borderBottomColor: t.border }]}
                onPress={() => toggleExpanded(key)}
                activeOpacity={0.7}
              >
                <Text style={[styles.tokenLabel, { color: t.text }]}>{tokenLabel}</Text>
                <View style={[styles.colorSwatch, { backgroundColor: currentColor, borderColor: t.border }]} />
              </TouchableOpacity>

              {isOpen && (
                <View style={[styles.swatchGrid, { backgroundColor: t.surface }]}>
                  {swatches[palette].map((hex) => {
                    const isActive = hex === currentColor;
                    return (
                      <TouchableOpacity
                        key={hex}
                        style={[styles.swatchCell, { backgroundColor: hex }]}
                        onPress={() => setColor(key, hex)}
                        accessibilityRole="button"
                        accessibilityState={{ selected: isActive }}
                        accessibilityLabel={hex}
                      >
                        {isActive && (
                          <Icon name="check" size={20} color={checkColorFor(hex)} strokeWidth={3} />
                        )}
                      </TouchableOpacity>
                    );
                  })}
                </View>
              )}
            </View>
          );
        })}

        {/* Save */}
        <TouchableOpacity
          style={[
            styles.saveBtn,
            { backgroundColor: label.trim().length > 0 ? t.accent : t.border },
          ]}
          onPress={handleSave}
          disabled={label.trim().length === 0}
          accessibilityRole="button"
          accessibilityLabel="Save theme"
        >
          <Text style={[styles.saveBtnText, { color: t.text }]}>Save</Text>
        </TouchableOpacity>

        {/* Delete (edit mode only) */}
        {mode === 'edit' && (
          <TouchableOpacity
            style={styles.deleteBtn}
            onPress={handleDelete}
            accessibilityRole="button"
            accessibilityLabel="Delete this theme"
          >
            <Text style={[styles.deleteBtnText, { color: t.textSecondary }]}>Delete theme</Text>
          </TouchableOpacity>
        )}

        <View style={styles.bottomPad} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, paddingTop: 60, paddingHorizontal: 20 },
  closeBtn: { position: 'absolute', top: 16, right: 16, padding: 8, zIndex: 10 },
  screenTitle: { fontSize: 22, fontWeight: '700', marginBottom: 20 },

  preview: {
    flexDirection: 'row',
    height: 48,
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 20,
  },
  previewSegment: { flex: 1 },

  nameInput: {
    borderRadius: 10,
    padding: 14,
    fontSize: 16,
    marginBottom: 20,
    borderWidth: 1,
  },

  tokenRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  tokenLabel: { fontSize: 15 },
  colorSwatch: { width: 28, height: 28, borderRadius: 14, borderWidth: 1 },

  swatchGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    padding: 12,
    borderRadius: 8,
    marginBottom: 4,
  },
  swatchCell: {
    width: 44,
    height: 44,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  swatchCheck: { fontSize: 16, fontWeight: '700' },

  saveBtn: {
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 12,
  },
  saveBtnText: { fontSize: 16, fontWeight: '700' },

  deleteBtn: { alignItems: 'center', paddingVertical: 12 },
  deleteBtnText: { fontSize: 14 },

  bottomPad: { height: 40 },
});
