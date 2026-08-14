import { Canvas, Path, Skia } from '@shopify/react-native-skia';
import React from 'react';
import { View } from 'react-native';

export interface DonutSegment {
  pct: number; // 0..1
  color: string;
}

export function DonutChart({
  segments,
  size = 84,
  strokeWidth = 14,
  holeColor,
}: {
  segments: DonutSegment[];
  size?: number;
  strokeWidth?: number;
  holeColor: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;
  let startAngle = -90;

  const arcs = segments.map((seg, i) => {
    const sweep = Math.max(seg.pct * 360, 0);
    const path = Skia.PathBuilder.Make()
      .addArc(
        { x: cx - radius, y: cy - radius, width: radius * 2, height: radius * 2 },
        startAngle,
        sweep,
      )
      .build();
    const el = { path, color: seg.color, key: i };
    startAngle += sweep;
    return el;
  });

  return (
    <View style={{ width: size, height: size }}>
      <Canvas style={{ width: size, height: size }}>
        {arcs.map(a => (
          <Path
            key={a.key}
            path={a.path}
            style="stroke"
            strokeWidth={strokeWidth}
            strokeCap="butt"
            color={a.color}
          />
        ))}
      </Canvas>
      {/* Inner hole to turn the stroked ring into a visual donut, matching
          the mockup's conic-gradient + masked center. */}
      <View
        style={{
          position: 'absolute',
          top: size / 2 - (radius - strokeWidth / 2) / 2,
          left: size / 2 - (radius - strokeWidth / 2) / 2,
          width: radius - strokeWidth / 2,
          height: radius - strokeWidth / 2,
          borderRadius: (radius - strokeWidth / 2) / 2,
          backgroundColor: holeColor,
        }}
      />
    </View>
  );
}
