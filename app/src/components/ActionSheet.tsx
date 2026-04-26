import { useState } from 'react';
import { Modal, Pressable, StyleSheet, Text } from 'react-native';
import { useTheme } from '../theme/useTheme';

export type SheetAction = { label: string; destructive?: boolean; onPress?: () => void };
export type SheetSpec = { title?: string; subtitle?: string; actions: SheetAction[] };

/** Bottom-anchored action sheet. Backdrop tap or explicit Cancel button dismisses. */
export function useActionSheet() {
  const [spec, setSpec] = useState<SheetSpec | null>(null);
  const close = () => setSpec(null);
  return {
    show: (s: SheetSpec) => setSpec(s),
    node: spec ? <Sheet spec={spec} onClose={close} /> : null,
  };
}

function Sheet({ spec, onClose }: { spec: SheetSpec; onClose: () => void }) {
  const t = useTheme();
  // Track which non-destructive action is the primary (first non-destructive). Subsequent
  // non-destructive actions get a neutral surfaceAlt fill so the primary stands out.
  let firstNonDestructiveSeen = false;
  return (
    <Modal visible transparent animationType="fade" hardwareAccelerated onRequestClose={onClose}>
      <Pressable style={s.backdrop} onPress={onClose}>
        <Pressable
          onPress={(e) => e.stopPropagation()}
          style={[s.card, { backgroundColor: t.surface, borderColor: t.border }]}
        >
          {spec.title ? <Text style={[s.title, { color: t.text }]}>{spec.title}</Text> : null}
          {spec.subtitle ? <Text style={[s.subtitle, { color: t.textSecondary }]}>{spec.subtitle}</Text> : null}
          {spec.actions.map((a, i) => {
            let bg: string;
            let textColor = '#fff';
            if (a.destructive) {
              bg = t.danger;
            } else if (!firstNonDestructiveSeen) {
              bg = t.accent;
              firstNonDestructiveSeen = true;
            } else {
              bg = t.surfaceAlt;
              textColor = t.text;
            }
            return (
              <Pressable
                key={i}
                onPress={() => { onClose(); a.onPress?.(); }}
                style={({ pressed }) => [s.btn, { backgroundColor: bg }, pressed && { opacity: 0.75 }]}
              >
                <Text style={[s.btnText, { color: textColor }]}>{a.label}</Text>
              </Pressable>
            );
          })}
          {/* Explicit Cancel — neutral surface-bordered, distinct from filled actions. */}
          <Pressable
            onPress={onClose}
            style={({ pressed }) => [
              s.btn,
              s.cancelBtn,
              { backgroundColor: t.surface, borderColor: t.border },
              pressed && { opacity: 0.75 },
            ]}
          >
            <Text style={[s.btnText, { color: t.text }]}>Cancel</Text>
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'flex-end' },
  card: { borderTopLeftRadius: 20, borderTopRightRadius: 20, borderWidth: 1, padding: 18, gap: 8 },
  title: { fontSize: 16, fontWeight: '700' },
  subtitle: { fontSize: 13 },
  btn: { borderRadius: 10, paddingVertical: 14, alignItems: 'center', marginTop: 4 },
  cancelBtn: { borderWidth: 1, marginTop: 8 },
  btnText: { fontWeight: '700', fontSize: 14 },
});
