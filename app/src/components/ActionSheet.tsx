import { useState } from 'react';
import { Modal, Pressable, StyleSheet, Text } from 'react-native';
import { useTheme } from '../theme/useTheme';

export type SheetAction = { label: string; destructive?: boolean; onPress?: () => void };
export type SheetSpec = { title?: string; subtitle?: string; actions: SheetAction[] };

/** Backdrop tap closes; no explicit Cancel button needed. */
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
  return (
    <Modal visible transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={s.backdrop} onPress={onClose}>
        <Pressable onPress={(e) => e.stopPropagation()} style={[s.card, { backgroundColor: t.surface, borderColor: t.border }]}>
          {spec.title ? <Text style={[s.title, { color: t.text }]}>{spec.title}</Text> : null}
          {spec.subtitle ? <Text style={[s.subtitle, { color: t.textSecondary }]}>{spec.subtitle}</Text> : null}
          {spec.actions.map((a, i) => (
            <Pressable
              key={i}
              onPress={() => { onClose(); a.onPress?.(); }}
              style={({ pressed }) => [s.btn, { backgroundColor: a.destructive ? t.danger : t.accent }, pressed && { opacity: 0.75 }]}
            >
              <Text style={s.btnText}>{a.label}</Text>
            </Pressable>
          ))}
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'center', paddingHorizontal: 24 },
  card: { borderRadius: 16, borderWidth: 1, padding: 18, gap: 8 },
  title: { fontSize: 16, fontWeight: '700' },
  subtitle: { fontSize: 13 },
  btn: { borderRadius: 10, paddingVertical: 12, alignItems: 'center', marginTop: 4 },
  btnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
});
