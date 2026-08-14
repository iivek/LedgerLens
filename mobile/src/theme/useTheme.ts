import { useColorScheme } from 'react-native';
import { darkColors, lightColors, type ColorTokens } from './tokens';

export interface Theme {
  colors: ColorTokens;
  isDark: boolean;
}

/** Single source of truth for light/dark colors -- mirrors the design
 * sketch's prefers-color-scheme handling, just resolved once per render
 * instead of via CSS custom properties. */
export function useTheme(): Theme {
  const scheme = useColorScheme();
  const isDark = scheme === 'dark';
  return { colors: isDark ? darkColors : lightColors, isDark };
}
