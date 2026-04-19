"""Cached list of real MTG set codes, fetched once from Scryfall and
pickled under /tmp. Used to filter OCR noise from plausible set codes."""
import json
import os
import pickle
from typing import Set

CACHE = '/tmp/scryfall_set_codes.pkl'


def known_set_codes() -> Set[str]:
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    import requests
    r = requests.get('https://api.scryfall.com/sets', timeout=10)
    r.raise_for_status()
    codes = {s['code'].lower() for s in r.json().get('data', [])}
    with open(CACHE, 'wb') as f:
        pickle.dump(codes, f)
    return codes
