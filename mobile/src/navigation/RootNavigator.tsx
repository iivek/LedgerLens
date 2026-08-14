import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import type { NavigatorScreenParams } from '@react-navigation/native';
import DashboardScreen from '../screens/DashboardScreen';
import ScanScreen from '../screens/ScanScreen';
import OverviewScreen from '../screens/OverviewScreen';
import SettingsScreen from '../screens/SettingsScreen';
import PrivacyScreen from '../screens/PrivacyScreen';
import SupportScreen from '../screens/SupportScreen';
import ReceiptDetailScreen from '../screens/ReceiptDetailScreen';
import { useTheme } from '../theme/useTheme';

export type SettingsStackParamList = {
  SettingsHome: undefined;
  Privacy: undefined;
  Support: undefined;
};

export type TabParamList = {
  Dashboard: undefined;
  Scan: undefined;
  Overview: undefined;
  SettingsStack: undefined;
};

// Root-level stack sitting above the tab bar so a receipt can be pushed full-
// screen (tabs hidden) from either Dashboard's "Recent" list or Overview's
// vendor list, without giving each tab its own duplicate detail stack.
export type RootStackParamList = {
  Tabs: NavigatorScreenParams<TabParamList>;
  ReceiptDetail: { id: string };
};

const Tab = createBottomTabNavigator<TabParamList>();
const SettingsStack = createNativeStackNavigator<SettingsStackParamList>();
const RootStack = createNativeStackNavigator<RootStackParamList>();

function SettingsStackNavigator() {
  return (
    <SettingsStack.Navigator screenOptions={{ headerShown: false }}>
      <SettingsStack.Screen name="SettingsHome" component={SettingsScreen} />
      <SettingsStack.Screen name="Privacy" component={PrivacyScreen} />
      <SettingsStack.Screen name="Support" component={SupportScreen} />
    </SettingsStack.Navigator>
  );
}

const TAB_ICON: Record<keyof TabParamList, string> = {
  Dashboard: '🏠',
  Scan: '📷',
  Overview: '📊',
  SettingsStack: '⚙️',
};

function MainTabs() {
  const { colors } = useTheme();
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: colors.moss,
        tabBarInactiveTintColor: colors.inkFaint,
        tabBarStyle: {
          backgroundColor: colors.paperRaised,
          borderTopColor: colors.line,
        },
        tabBarIcon: () => null,
        // Plain string, not JSX -- bottom-tabs wraps a returned string in its
        // own styled <Text> automatically; returning a bare text Fragment
        // instead would throw ("Text strings must be rendered within a
        // <Text> component"), since RN doesn't auto-wrap raw text nodes.
        tabBarLabel: `${TAB_ICON[route.name as keyof TabParamList]} ${
          route.name === 'SettingsStack' ? 'Settings' : route.name
        }`,
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Scan" component={ScanScreen} />
      <Tab.Screen name="Overview" component={OverviewScreen} />
      <Tab.Screen
        name="SettingsStack"
        component={SettingsStackNavigator}
        options={{ title: 'Settings' }}
      />
    </Tab.Navigator>
  );
}

// Root stack sits above the tab bar so ReceiptDetail renders full-screen
// (tabs hidden) when pushed from either Dashboard or Overview.
export default function RootNavigator() {
  return (
    <RootStack.Navigator screenOptions={{ headerShown: false }}>
      <RootStack.Screen name="Tabs" component={MainTabs} />
      <RootStack.Screen
        name="ReceiptDetail"
        component={ReceiptDetailScreen}
        options={{ presentation: 'card' }}
      />
    </RootStack.Navigator>
  );
}
