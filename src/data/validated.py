"""Offset-preserving CSV audit and deterministic note-level evaluation splits."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from .synthetic_generator import ClinicalSample


def load_validated(root):
    root = Path(root)
    paths = [root / 'train_notes.csv', root / 'train_annotations.csv']
    notes = pd.read_csv(paths[0], dtype={'note_id': str}, keep_default_na=False)
    annotations = pd.read_csv(paths[1], dtype={'note_id': str}, keep_default_na=False)
    if notes.note_id.duplicated().any():
        raise ValueError('Duplicate note IDs; resolve before evaluation.')
    texts = dict(zip(notes.note_id, notes.text))
    grouped = {nid: [] for nid in texts}
    counts, rejected, seen = Counter(), [], set()
    labels = {x.lower(): x for x in ['Problem', 'Treatment', 'Test']}
    for idx, row in annotations.iterrows():
        reason = None
        label = labels.get(str(row.entity_type).strip().lower())
        try:
            sf, ef = float(row.start), float(row.end)
            s, e = int(sf), int(ef)
            if sf != s or ef != e:
                raise ValueError()
        except (ValueError, OverflowError):
            reason = 'invalid_offsets'
        text = texts.get(row.note_id)
        if reason is None:
            if text is None: reason = 'missing_note'
            elif label is None: reason = 'unknown_label'
            elif not 0 <= s < e <= len(text): reason = 'invalid_bounds'
            elif text[s:e] != row.span: reason = 'text_mismatch'
            elif (row.note_id, s, e, label) in seen: reason = 'duplicate'
        if reason:
            counts[reason] += 1
            rejected.append({'csv_row': int(idx) + 2, 'note_id': row.note_id, 'reason': reason})
            continue
        seen.add((row.note_id, s, e, label))
        grouped[row.note_id].append(dict(start=s, end=e, text=text[s:e], label=label))
        counts[label] += 1
    samples = [ClinicalSample(text=t, spans=grouped[n], note_id=n, source='csv') for n,t in texts.items()]
    audit = {'notes': len(samples), 'input_annotations': len(annotations), 'counts': dict(counts),
             'rejected': rejected, 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    return samples, audit


def split_notes(samples, seed=42):
    """Group identical note text as well as note IDs; no entity-label-based split search."""
    groups = {}
    for s in samples:
        key = hashlib.sha256(s.text.encode()).hexdigest()
        groups.setdefault(key, []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    a, b = int(n * .6), int(n * .8)
    if not 0 < a < b < n:
        raise ValueError('Need at least five distinct notes for support/dev/test splits.')
    return {name: [s for key in ks for s in groups[key]] for name,ks in
            [('support', keys[:a]), ('dev', keys[a:b]), ('test', keys[b:])]}
