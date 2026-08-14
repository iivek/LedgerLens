import { Platform } from 'react-native';

// Same palette/typography system as the design sketch: pale "greenbar" ledger
// paper rather than generic fintech gradients or cafe-cream + terracotta.
// Kept in one place (not scattered inline styles) so light/dark stay in sync.

export const lightColors = {
  paper: '#EEF1E6',
  paperRaised: '#F7F8F2',
  paperSunken: '#E4E8D9',
  ink: '#1B2420',
  inkSoft: '#4B564E',
  inkFaint: '#7C8A7E',
  moss: '#3F6B4F',
  mossDeep: '#274736',
  mossTint: '#DCE6DA',
  ochre: '#A9762C',
  oxblood: '#7A3428',
  line: '#D7DBC9',
  lineStrong: '#C1C9AF',
} as const;

export const darkColors = {
  paper: '#141C16',
  paperRaised: '#1C271E',
  paperSunken: '#0F1712',
  ink: '#E9EFE4',
  inkSoft: '#AAB9A9',
  inkFaint: '#718274',
  moss: '#7CB595',
  mossDeep: '#4A8067',
  mossTint: '#24352A',
  ochre: '#D9AA5E',
  oxblood: '#CE8571',
  line: '#2B3A2E',
  lineStrong: '#3A4C3D',
} as const;

// Deliberately theme-invariant (see the design sketch's .commitments banner) --
// reads identically in light and dark, used only for the fixed "stamp" surface.
export const stamp = {
  bg: '#24382C',
  text: '#EFF3E6',
} as const;

// Widen to plain `string` values -- `typeof lightColors` alone would infer
// each hex code as its own string-literal type (via the `as const` above),
// which then rejects darkColors (different literal hex values) as an
// incompatible type despite having the identical key shape.
export type ColorTokens = { [K in keyof typeof lightColors]: string };

export const fonts = {
  // "Georgia"/"Palatino" are real built-in faces on iOS; Android has no
  // equivalent installed by default, so it falls back to the generic serif
  // family (typically Noto Serif) -- both read as a bookish, ledger-heading
  // face rather than a generic sans, which is what this needs to carry.
  display: Platform.select({ ios: 'Georgia', android: 'serif', default: 'serif' }),
  body: Platform.select({
    ios: 'System',
    android: 'sans-serif',
    default: 'System',
  }),
  mono: Platform.select({
    ios: 'Menlo',
    android: 'monospace',
    default: 'monospace',
  }),
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  sm: 8,
  md: 14,
  lg: 22,
  pill: 999,
} as const;
