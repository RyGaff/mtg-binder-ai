export type SwatchCategory = 'background' | 'surface' | 'border' | 'text' | 'textSecondary' | 'accent';

export const swatches: Record<SwatchCategory, string[]> = {
  background: [
    '#000000', '#0a0a0a', '#111318', '#0d1117',
    '#0f1923', '#1a0d1f', '#0d1a0d', '#1a1c23',
    '#252830', '#2d3142', '#3a3a3a', '#4a4a4a',
    '#666666', '#888888', '#aaaaaa', '#cccccc',
    '#e0e0e0', '#e5e5ea', '#f0f0f0', '#f2f2f7',
    '#f5f5f5', '#f8f8f8', '#fafafa', '#ffffff',
  ],
  surface: [
    '#0d0d0d', '#141414', '#1a1c23', '#1e2028',
    '#252830', '#2a2d38', '#303340', '#383b47',
    '#404450', '#4a4e5c', '#555555', '#666666',
    '#777777', '#888888', '#999999', '#aaaaaa',
    '#cccccc', '#d8d8d8', '#e0e0e0', '#e5e5ea',
    '#efefef', '#f5f5f5', '#fafafa', '#ffffff',
  ],
  border: [
    '#1a1a1a', '#222222', '#2a2d38', '#333333',
    '#3d3d3d', '#444444', '#4d4d4d', '#555555',
    '#5e5e5e', '#666666', '#707070', '#7c7c7c',
    '#888888', '#999999', '#aaaaaa', '#b0b0b0',
    '#bbbbbb', '#c7c7cc', '#cccccc', '#d0d0d0',
    '#d8d8d8', '#dddddd', '#e0e0e0', '#e8e8e8',
  ],
  text: [
    '#ffffff', '#f8f8f8', '#f0f0f0', '#e8e8e8',
    '#dddddd', '#cccccc', '#bbbbbb', '#aaaaaa',
    '#999999', '#888888', '#777777', '#666666',
    '#555555', '#444444', '#333333', '#222222',
    '#1a1a1a', '#111111', '#0a0a0a', '#000000',
    '#e8e0d0', '#d0e8d0', '#d0d0e8', '#e8d0d0',
  ],
  textSecondary: [
    '#888888', '#8a8a99', '#8899aa', '#99aa88',
    '#aa8888', '#aa88aa', '#88aa88', '#88aaaa',
    '#6d6d72', '#777777', '#666666', '#555555',
    '#7a7a8a', '#8a7a7a', '#7a8a7a', '#7a8a8a',
    '#999999', '#aaaaaa', '#a0a0b0', '#b0a0a0',
    '#a0b0a0', '#a0b0b0', '#b0a0b0', '#a8a8a8',
  ],
  accent: [
    '#4ecdc4', '#26c6da', '#29b6f6', '#42a5f5',
    '#5c6bc0', '#7e57c2', '#ab47bc', '#ec407a',
    '#ef5350', '#ff7043', '#ffa726', '#ffca28',
    '#d4e157', '#9ccc65', '#66bb6a', '#26a69a',
    '#00bcd4', '#039be5', '#1e88e5', '#3949ab',
    '#8e24aa', '#e91e63', '#f44336', '#ff5722',
  ],
};
