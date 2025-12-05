import { MaterialIcons } from '@expo/vector-icons'; // We use this for the Camera/History icons
import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';

import { HapticTab } from '@/components/haptic-tab';
import { useColorScheme } from '@/hooks/use-color-scheme';

// --- CUSTOM RICE DOCTOR THEME COLORS ---
const COLORS = {
  primary: '#2E7D32',    // Deep Green
  inactive: '#9E9E9E',   // Grey
  background: '#FFFFFF', // White
};

export default function TabLayout() {
  const colorScheme = useColorScheme();

  return (
    <Tabs
      screenOptions={{
        // 1. Color Settings
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.inactive,
        
        // 2. Hide the top header (because our screens have their own green headers)
        headerShown: false,
        
        // 3. Keep your Haptic feedback button
        tabBarButton: HapticTab,

        // 4. Style the Bar (Taller and cleaner)
        tabBarStyle: {
          backgroundColor: COLORS.background,
          borderTopWidth: 0, 
          elevation: 5, // Shadow for Android
          height: Platform.OS === 'ios' ? 85 : 65, 
          paddingBottom: Platform.OS === 'ios' ? 25 : 10,
          paddingTop: 10,
        },
        tabBarLabelStyle: {
          fontWeight: 'bold',
          fontSize: 12,
        }
      }}>

      {/* --- TAB 1: SCANNER (Home) --- */}
      <Tabs.Screen
        name="index"
        options={{
          title: 'Scanner',
          tabBarIcon: ({ color }) => (
            // Using MaterialIcons for the Camera
            <MaterialIcons size={28} name="camera-alt" color={color} />
          ),
        }}
      />

      {/* --- TAB 2: HISTORY (Explore) --- */}
      <Tabs.Screen
        name="explore"
        options={{
          title: 'History',
          tabBarIcon: ({ color }) => (
            // Using MaterialIcons for the History clock
            <MaterialIcons size={28} name="history" color={color} />
          ),
        }}
      />
    </Tabs>
  );
}