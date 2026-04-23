import { Tabs, useRouter } from 'expo-router';
import { TouchableOpacity } from 'react-native';
import { useTheme } from '../../src/theme/useTheme';
import { spacing, HIT_SLOP_8 } from '../../src/theme/themes';
import { Icon, type IconName } from '../../src/components/icons/Icon';

function tabIcon(name: IconName) {
  return ({ color, size }: { color: string; size: number }) => (
    <Icon name={name} size={size} color={color} strokeWidth={1.8} />
  );
}

export default function TabLayout() {
  const router = useRouter();
  const theme = useTheme();

  return (
    <Tabs
      screenOptions={{
        tabBarStyle: { backgroundColor: theme.surface, borderTopColor: theme.border },
        tabBarActiveTintColor: theme.accent,
        tabBarInactiveTintColor: theme.textSecondary,
        headerStyle: { backgroundColor: theme.surface },
        headerTintColor: theme.text,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Binder',
          tabBarIcon: tabIcon('binder'),
          tabBarAccessibilityLabel: 'Binder',
          headerRight: () => (
            <TouchableOpacity
              onPress={() => router.push('/profile')}
              style={{ marginRight: spacing.lg, padding: spacing.sm }}
              hitSlop={HIT_SLOP_8}
              accessibilityRole="button"
              accessibilityLabel="Profile"
            >
              <Icon name="profile" size={22} color={theme.text} />
            </TouchableOpacity>
          ),
        }}
      />
      <Tabs.Screen
        name="search"
        options={{
          title: 'Search',
          tabBarIcon: tabIcon('search'),
          tabBarAccessibilityLabel: 'Search',
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: 'Scan',
          tabBarIcon: tabIcon('camera'),
          tabBarAccessibilityLabel: 'Scan',
        }}
      />
      <Tabs.Screen
        name="decks"
        options={{
          title: 'Decks',
          tabBarIcon: tabIcon('cards'),
          tabBarAccessibilityLabel: 'Decks',
        }}
      />
    </Tabs>
  );
}
