import type { CollectionEntryWithCard } from '../db/collection';

export type ImportRow = {
  scryfall_id: string;
  name: string;
  quantity: number;
  foil: boolean;
  condition: string;
  set_code?: string;
  collector_number?: string;
};

export function serializeToJson(entries: CollectionEntryWithCard[]): string {
  return JSON.stringify(
    entries.map((e) => ({
      scryfall_id: e.scryfall_id,
      name: e.name,
      set_code: e.set_code,
      collector_number: e.collector_number,
      quantity: e.quantity,
      foil: e.foil,
      condition: e.condition,
    })),
    null,
    2
  );
}

const CSV_HEADERS = [
  'scryfall_id',
  'name',
  'set_code',
  'collector_number',
  'quantity',
  'foil',
  'condition',
];

export function serializeToCsv(entries: CollectionEntryWithCard[]): string {
  const rows = entries.map((e) =>
    [
      e.scryfall_id,
      e.name,
      e.set_code,
      e.collector_number,
      e.quantity,
      e.foil ? '1' : '0',
      e.condition,
    ]
      .map((v) => `"${String(v).replace(/"/g, '""')}"`)
      .join(',')
  );
  return [CSV_HEADERS.join(','), ...rows].join('\n');
}

// Maps Moxfield condition strings to app condition codes
function mapCondition(raw: string): string {
  switch (raw.trim()) {
    case 'Near Mint': return 'NM';
    case 'Lightly Played': return 'LP';
    case 'Moderately Played': return 'MP';
    case 'Heavily Played': return 'HP';
    case 'Damaged': return 'DMG';
    default: return 'NM';
  }
}

function parseRow(obj: Record<string, string>): ImportRow {
  // Moxfield format detection: has "Count" and "Edition" columns
  if ('Count' in obj) {
    return {
      scryfall_id: '',
      name: obj.Name ?? '',
      quantity: parseInt(obj.Count ?? '1', 10),
      foil: obj.Foil === 'foil' || obj.Foil === 'etched',
      condition: mapCondition(obj.Condition ?? 'Near Mint'),
      set_code: obj.Edition?.toLowerCase() ?? '',
      collector_number: obj['Collector Number'] ?? '',
    };
  }
  // App's own CSV format
  return {
    scryfall_id: obj.scryfall_id ?? '',
    name: obj.name ?? '',
    quantity: parseInt(obj.quantity ?? '1', 10),
    foil: obj.foil === '1',
    condition: obj.condition ?? 'NM',
    set_code: obj.set_code ?? '',
    collector_number: obj.collector_number ?? '',
  };
}

/** Parse a single CSV line respecting quoted fields that may contain commas. */
function splitCsvLine(line: string): string[] {
  const fields: string[] = [];
  let i = 0;
  while (i < line.length) {
    if (line[i] === '"') {
      // Quoted field
      let field = '';
      i++; // skip opening quote
      while (i < line.length) {
        if (line[i] === '"' && line[i + 1] === '"') {
          field += '"';
          i += 2;
        } else if (line[i] === '"') {
          i++; // skip closing quote
          break;
        } else {
          field += line[i++];
        }
      }
      fields.push(field);
      if (line[i] === ',') i++; // skip comma separator
    } else {
      // Unquoted field
      const end = line.indexOf(',', i);
      if (end === -1) {
        fields.push(line.slice(i));
        break;
      }
      fields.push(line.slice(i, end));
      i = end + 1;
    }
  }
  return fields;
}

export function parseImportFile(
  content: string,
  format: 'json' | 'csv'
): ImportRow[] {
  if (format === 'json') {
    return JSON.parse(content) as ImportRow[];
  }
  const lines = content.split('\n').filter(Boolean);
  const headers = splitCsvLine(lines[0]).map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => {
      obj[h] = values[i] ?? '';
    });
    return parseRow(obj);
  });
}
