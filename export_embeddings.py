"""
Converts Embedded_Magic_cards.pkl → embeddings.bin

Binary format:
  Header:  [uint32 N][uint32 D]  (little-endian)
  Records: [36-byte ASCII scryfall_id][64-byte card name, NUL-padded][D × float32 LE]  × N

The 64-byte name field lets the app resolve a different printing of the same card
to its embedding when the scryfall_id doesn't match directly.

Usage:
  python export_embeddings.py
  python export_embeddings.py --pkl Embedded_Magic_cards.pkl --out embeddings.bin
"""

import argparse
import json
import struct
import numpy as np
import pandas as pd


def load_name_to_scryfall_id(oracle_cards_path: str) -> dict[str, str]:
    with open(oracle_cards_path, "r") as f:
        data = json.load(f)
    mapping = {}
    for card in data:
        name = card.get("name", "")
        scryfall_id = card.get("id", "")
        if name and scryfall_id and name not in mapping:
            mapping[name] = scryfall_id
    return mapping


def export(pkl_path: str, oracle_cards_path: str, out_path: str) -> None:
    print(f"Loading embeddings from {pkl_path}...")
    df = pd.read_pickle(pkl_path)

    print(f"Loading name → scryfall_id map from {oracle_cards_path}...")
    name_to_id = load_name_to_scryfall_id(oracle_cards_path)

    records = []
    missing = []
    for card_name, row in df.iterrows():
        scryfall_id = name_to_id.get(str(card_name))
        if not scryfall_id or len(scryfall_id) != 36:
            missing.append(str(card_name))
            continue
        embedding = np.array(row["embeddings"], dtype=np.float32)
        records.append((scryfall_id, str(card_name), embedding))

    if missing:
        print(f"  Skipped {len(missing)} cards with no scryfall_id match")
        if len(missing) <= 10:
            for m in missing:
                print(f"    - {m}")

    if not records:
        raise RuntimeError("No records to write — check that pkl and oracle-cards JSON match")

    n = len(records)
    d = len(records[0][2])
    print(f"Writing {n} cards, dimension {d} → {out_path}")

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<II", n, d))
        # Records
        for scryfall_id, card_name, embedding in records:
            id_bytes = scryfall_id.encode("ascii")
            assert len(id_bytes) == 36, f"UUID must be 36 bytes, got {len(id_bytes)}"
            name_bytes = card_name.encode("utf-8")[:64].ljust(64, b"\0")
            f.write(id_bytes)
            f.write(name_bytes)
            f.write(embedding.tobytes())

    size_mb = (8 + n * (36 + 64 + d * 4)) / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="Embedded_Magic_cards.pkl")
    parser.add_argument("--cards", default="Data/oracle-cards-.json")
    parser.add_argument("--out", default="embeddings.bin")
    args = parser.parse_args()

    export(args.pkl, args.cards, args.out)
