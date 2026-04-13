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
import { useTheme } from '../src/theme/useTheme';
import { themes, type CustomTheme, type CustomThemeName } from '../src/theme/themes';
import { swatches } from '../src/theme/swatches';

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
  const { setCustomTheme, deleteCustomTheme, setTheme, customThemes, theme } = useStore();
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
    if (!isValidSlot) router.back();
  }, [isValidSlot]);

  if (!isValidSlot) return null;

  const setColor = (key: TokenKey, value: string) => {
    setColors((prev) => ({ ...prev, [key]: value }));
  };

  const toggleExpanded = (key: TokenKey) => {
    setExpandedToken((prev) => (prev === key ? null : key));
  };

  const handleSave = () => {
    const custom: CustomTheme = { name: customName, label, ...colors };
    setCustomTheme(slot, custom);
    if (mode === 'new' || theme === customName) {
      setTheme(customName);
    }
    router.back();
  };

  const handleDelete = () => {
    if (theme === customName) setTheme('dark');
    deleteCustomTheme(slot);
    router.back();
  };

  return (
    <View style={[styles.screen, { backgroundColor: t.bg }]}>
      {/* Close button */}
      <TouchableOpacity
        style={styles.closeBtn}
        onPress={() => router.back()}
        accessibilityLabel="Close without saving"
        accessibilityRole="button"
      >
        <Text style={[styles.closeBtnText, { color: t.textSecondary }]}>✕</Text>
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
                        accessibilityLabel={hex}
                      >
                        {isActive && (
                          <Text style={styles.swatchCheck}>✓</Text>
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
  closeBtnText: { fontSize: 18 },
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
    width: 36,
    height: 36,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  swatchCheck: { color: '#ffffff', fontSize: 16, fontWeight: '700', textShadowColor: '#000', textShadowRadius: 2, textShadowOffset: { width: 0, height: 1 } },

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
