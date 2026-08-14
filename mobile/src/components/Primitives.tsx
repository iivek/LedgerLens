import React from 'react';
import { StyleSheet, Text, View, type ViewStyle } from 'react-native';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

export function Card({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: ViewStyle;
}) {
  const { colors } = useTheme();
  return (
    <View
      style={[
        styles.card,
        { backgroundColor: colors.paper, borderColor: colors.line },
        style,
      ]}
    >
      {children}
    </View>
  );
}

/** Tabular-aligned money figure. Monospace gives fixed-width digits on both
 * platforms even where the fontVariant tabular-nums CSS hint isn't honored. */
export function Money({
  value,
  currency = '$',
  size = 14,
  emphasis = false,
}: {
  value: number;
  currency?: string;
  size?: number;
  emphasis?: boolean;
}) {
  const { colors } = useTheme();
  return (
    <Text
      style={{
        fontFamily: fonts.mono,
        fontSize: size,
        fontWeight: emphasis ? '700' : '400',
        color: colors.ink,
      }}
    >
      {currency}
      {value.toFixed(2)}
    </Text>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  const { colors } = useTheme();
  return (
    <Text style={[styles.sectionLabel, { color: colors.inkFaint }]}>
      {children}
    </Text>
  );
}

export function LocalBadge({ label }: { label: string }) {
  const { colors } = useTheme();
  return (
    <View style={[styles.badge, { backgroundColor: colors.mossTint }]}>
      <View style={[styles.dot, { backgroundColor: colors.moss }]} />
      <Text style={[styles.badgeText, { color: colors.mossDeep }]}>
        {label}
      </Text>
    </View>
  );
}

export function EmptyState({
  icon,
  title,
  body,
  actionLabel,
  onAction,
}: {
  icon: string;
  title: string;
  body: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  const { colors } = useTheme();
  return (
    <View style={styles.emptyState}>
      <Text style={{ fontSize: 34, marginBottom: spacing.sm }}>{icon}</Text>
      <Text
        style={{
          color: colors.ink,
          fontFamily: fonts.display,
          fontSize: 17,
          marginBottom: 6,
          textAlign: 'center',
        }}
      >
        {title}
      </Text>
      <Text
        style={{
          color: colors.inkFaint,
          fontSize: 12.5,
          textAlign: 'center',
          lineHeight: 18,
          maxWidth: 260,
        }}
      >
        {body}
      </Text>
      {actionLabel && onAction && (
        <Text
          onPress={onAction}
          style={[
            styles.emptyAction,
            { backgroundColor: colors.moss, color: colors.paperRaised },
          ]}
        >
          {actionLabel}
        </Text>
      )}
    </View>
  );
}

export function CategoryChip({
  label,
  icon,
  selected,
  onPress,
}: {
  label: string;
  icon?: string;
  selected?: boolean;
  onPress?: () => void;
}) {
  const { colors } = useTheme();
  return (
    <Text
      onPress={onPress}
      style={[
        styles.chip,
        {
          backgroundColor: selected ? colors.moss : colors.paperRaised,
          borderColor: selected ? colors.moss : colors.lineStrong,
          color: selected ? colors.paperRaised : colors.inkSoft,
          fontWeight: selected ? '700' : '400',
        },
      ]}
    >
      {icon ? `${icon} ` : ''}
      {label}
    </Text>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: radius.md,
    borderWidth: 1,
    padding: spacing.md,
  },
  sectionLabel: {
    fontSize: 11.5,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: spacing.sm,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
    gap: 6,
  },
  dot: { width: 5, height: 5, borderRadius: 3 },
  badgeText: { fontSize: 10.5, letterSpacing: 0.3 },
  chip: {
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 6,
    fontSize: 12,
    overflow: 'hidden',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xl * 1.5,
    paddingHorizontal: spacing.lg,
  },
  emptyAction: {
    marginTop: spacing.lg,
    borderRadius: radius.pill,
    paddingHorizontal: 20,
    paddingVertical: 10,
    fontSize: 12.5,
    fontWeight: '700',
    overflow: 'hidden',
  },
});
