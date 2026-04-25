import { memo } from 'react';
import Svg, { Path, Circle, Rect, Line } from 'react-native-svg';

export type IconName =
  | 'binder' | 'search' | 'camera' | 'cards' | 'profile' | 'close' | 'check'
  | 'plus' | 'sparkle' | 'sparkle-outline' | 'clock' | 'pencil' | 'chevron-right';

type Props = { name: IconName; size?: number; color?: string; strokeWidth?: number };

function IconImpl({ name, size = 24, color = '#fff', strokeWidth = 2 }: Props) {
  const common = {
    width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: color,
    strokeWidth, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const,
  };

  switch (name) {
    case 'binder':
      return (
        <Svg {...common}>
          <Rect x={3} y={4} width={18} height={16} rx={2} />
          <Line x1={3} y1={9} x2={21} y2={9} />
          <Line x1={8} y1={4} x2={8} y2={20} />
        </Svg>
      );
    case 'search':
      return (
        <Svg {...common}>
          <Circle cx={11} cy={11} r={7} />
          <Line x1={16.5} y1={16.5} x2={21} y2={21} />
        </Svg>
      );
    case 'camera':
      return (
        <Svg {...common}>
          <Path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" />
          <Circle cx={12} cy={13} r={3.5} />
        </Svg>
      );
    case 'cards':
      return (
        <Svg {...common}>
          <Rect x={3} y={6} width={12} height={15} rx={2} transform="rotate(-8 9 13.5)" />
          <Rect x={9} y={3} width={12} height={15} rx={2} transform="rotate(8 15 10.5)" />
        </Svg>
      );
    case 'profile':
      return (
        <Svg {...common}>
          <Circle cx={12} cy={8} r={4} />
          <Path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8" />
        </Svg>
      );
    case 'close':
      return (
        <Svg {...common}>
          <Line x1={6} y1={6} x2={18} y2={18} />
          <Line x1={18} y1={6} x2={6} y2={18} />
        </Svg>
      );
    case 'check':
      return <Svg {...common}><Path d="M5 12.5l4.5 4.5L19 7.5" /></Svg>;
    case 'plus':
      return (
        <Svg {...common}>
          <Line x1={12} y1={5} x2={12} y2={19} />
          <Line x1={5} y1={12} x2={19} y2={12} />
        </Svg>
      );
    case 'sparkle':
      return (
        <Svg {...common} fill={color} stroke="none">
          <Path d="M12 2 13.6 9 21 12 13.6 15 12 22 10.4 15 3 12 10.4 9z" />
        </Svg>
      );
    case 'sparkle-outline':
      return <Svg {...common}><Path d="M12 2 13.6 9 21 12 13.6 15 12 22 10.4 15 3 12 10.4 9z" /></Svg>;
    case 'clock':
      return (
        <Svg {...common}>
          <Circle cx={12} cy={12} r={9} />
          <Path d="M12 7v5l3.5 2" />
        </Svg>
      );
    case 'pencil':
      return (
        <Svg {...common}>
          <Path d="M4 20h4l10-10-4-4L4 16v4z" />
          <Line x1={13.5} y1={6.5} x2={17.5} y2={10.5} />
        </Svg>
      );
    case 'chevron-right':
      return <Svg {...common}><Path d="M9 6l6 6-6 6" /></Svg>;
  }
}

export const Icon = memo(IconImpl);
