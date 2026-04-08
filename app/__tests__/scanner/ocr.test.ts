import { parseSetAndNumber } from '../../src/scanner/ocr';

describe('parseSetAndNumber', () => {
  it('parses a standard set+number string', () => {
    const result = parseSetAndNumber('lea 161/302');
    expect(result).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });

  it('parses without slash (just number)', () => {
    const result = parseSetAndNumber('m21 420');
    expect(result).toEqual({ setCode: 'm21', collectorNumber: '420' });
  });

  it('parses 3-char set codes', () => {
    const result = parseSetAndNumber('cmr 085/361');
    expect(result).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });

  it('returns null for unrecognizable text', () => {
    const result = parseSetAndNumber('not a card');
    expect(result).toBeNull();
  });

  it('handles mixed case set codes', () => {
    const result = parseSetAndNumber('LEA 161');
    expect(result).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
});
