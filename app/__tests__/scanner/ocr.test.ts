import { parseSetAndNumber } from '../../src/scanner/ocr';

describe('parseSetAndNumber', () => {
  // Legacy format: set code before number
  it('parses set-code-first format', () => {
    expect(parseSetAndNumber('lea 161/302')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });

  it('parses set code without slash', () => {
    expect(parseSetAndNumber('m21 420')).toEqual({ setCode: 'm21', collectorNumber: '420' });
  });

  it('parses 3-char set codes', () => {
    expect(parseSetAndNumber('cmr 085/361')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });

  it('handles mixed case', () => {
    expect(parseSetAndNumber('LEA 161')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });

  // Modern format: number before set code (real card bottom-left corner)
  it('parses modern card format: number rarity set lang', () => {
    expect(parseSetAndNumber('042/350 R IKO EN')).toEqual({ setCode: 'iko', collectorNumber: '042' });
  });

  it('parses modern format without total', () => {
    expect(parseSetAndNumber('161 R LEA EN')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });

  it('ignores language and rarity tokens', () => {
    expect(parseSetAndNumber('085/361 C CMR EN')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });

  it('handles OCR noise around the real data', () => {
    expect(parseSetAndNumber('Illustrated by John Avon\n085/361 C CMR EN')).toEqual({
      setCode: 'cmr',
      collectorNumber: '085',
    });
  });

  it('returns null for unrecognizable text', () => {
    expect(parseSetAndNumber('not a card')).toBeNull();
  });

  it('returns null when no collector number present', () => {
    expect(parseSetAndNumber('IKO EN R')).toBeNull();
  });
});
