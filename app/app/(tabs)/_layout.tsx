import { Tabs, useRouter} from 'expo-router';
import { Text, TouchableOpacity } from 'react-native';

function TabIcon({ label }: { label: string }) {
  return <Text style={{ fontSize: 18 }}>{label}</Text>;
}

export default function TabLayout() {
  const router = useRouter();

  return (
    <Tabs
      screenOptions={{
        tabBarStyle: { backgroundColor: '#111318', borderTopColor: '#7c848e' },
        tabBarActiveTintColor: '#4ecdc4',
        tabBarInactiveTintColor: '#888',
        headerStyle: { backgroundColor: '#30343f' },
        headerTintColor: '#fff',
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Binder',
          tabBarIcon: () => <TabIcon label="📦" />,
          headerRight: () => (
            <TouchableOpacity
              onPress={() => router.push('/profile')}
              style={{ marginRight: 16 }}
              accessibilityLabel='Profile'
              >
                <Text style={{ fontSize: 20 }}>👤</Text>
              </TouchableOpacity>
          )
        }}
      />
      <Tabs.Screen
        name="search"
        options={{
          title: 'Search',
          tabBarIcon: () => <TabIcon label="🔍" />,
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: 'Scan',
          tabBarIcon: () => <TabIcon label="📷" />,
        }}
      />
      <Tabs.Screen
        name="decks"
        options={{
          title: 'Decks',
          tabBarIcon: () => <TabIcon label="🃏" />,
        }}
      />
    </Tabs>
  );
}
