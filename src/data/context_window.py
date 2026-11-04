def centered_window(offsets, start, end, context_tokens, max_tokens):
    hits = [i for i,(s,e) in enumerate(offsets) if s < end and e > start and e > s]
    if not hits:
        raise ValueError('Mention has no overlapping tokens.')
    first, last = hits[0], hits[-1] + 1
    if offsets[first][0] > start or offsets[last-1][1] < end:
        raise ValueError('Tokenizer does not fully cover mention offsets.')
    width = last - first
    if width > max_tokens:
        raise ValueError('Mention itself exceeds model token budget; no silent truncation.')
    extra = min(context_tokens, (max_tokens-width)//2)
    lo, hi = max(0, first-extra), min(len(offsets), last+extra)
    return lo, hi, first-lo, last-lo

