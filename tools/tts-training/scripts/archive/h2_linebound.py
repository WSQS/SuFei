"""Shared line-boundary computation for H2 training and evaluation.

Extracted from h2_train.py to avoid importing wandb in eval context.
"""
import numpy as np

from generate_paddlespeech_distillation_data import (
    text_to_phonemes,
    PUNCT_TO_PHONE,
)

DELIM_CHARS = set("\uff0c\u3002\uff1f\uff01\uff1b\uff1a")


def compute_body_lines(text, phoneme_ids, durations):
    """Identify body line segments at phoneme/frame granularity.

    Returns list of segments:
      {phone_start, phone_end (exclusive), frame_start, frame_end (exclusive),
       n_phones, n_frames, text}
    Excludes header (title/dynasty/author) and <eos>.
    """
    n_phones = len(phoneme_ids)
    phones = text_to_phonemes(text)
    assert len(phones) == n_phones, (
        "Phone mismatch %s: g2p=%d vs manifest=%d"
        % (text[:20], len(phones), n_phonemes)
    )
    cumsum = [0] + list(np.cumsum(durations))

    # --- locate body start in text ---
    dot = text.find("\u00b7")  # middle dot
    if dot >= 0:
        period = text.find("\u3002", dot)  # period after middle dot
        body_start_char = period + 1 if period >= 0 else len(text)
    else:
        fp = text.find("\u3002")
        body_start_char = fp + 1 if fp >= 0 else 0

    # --- find delimiter positions in body ---
    body_delims = []
    for i in range(body_start_char, len(text)):
        if text[i] in DELIM_CHARS:
            body_delims.append(i)

    if not body_delims:
        return []

    # --- split body into lines at delimiters ---
    raw_lines = []
    prev = body_start_char
    for di in body_delims:
        raw_lines.append((prev, di + 1))
        prev = di + 1

    # --- map text positions to phone indices ---
    char_phone_range = {}
    pi = 0
    for ti, char in enumerate(text):
        is_cjk = "\u4e00" <= char <= "\u9fff"
        is_punct = char in PUNCT_TO_PHONE
        cap = 2 if is_cjk else (1 if is_punct else 0)
        start_pi = pi
        cnt = 0
        while cnt < cap and pi < n_phones and phones[pi][1] is not None \
                and phones[pi][1] == char:
            cnt += 1
            pi += 1
        char_phone_range[ti] = (start_pi, pi)

    # --- convert text-line ranges to phone/frame segments ---
    segments = []
    for ts, te in raw_lines:
        p_start = None
        p_end = None
        for ti in range(ts, te):
            if ti in char_phone_range:
                ps, pe = char_phone_range[ti]
                if ps < pe:
                    if p_start is None:
                        p_start = ps
                    p_end = pe
        if p_start is None or p_end is None or p_end <= p_start:
            continue
        f_start = cumsum[p_start]
        f_end = cumsum[min(p_end, n_phones)]
        n_fr = f_end - f_start
        if n_fr < 10:
            continue
        segments.append({
            "phone_start": p_start,
            "phone_end": p_end,
            "frame_start": f_start,
            "frame_end": f_end,
            "n_phones": p_end - p_start,
            "n_frames": n_fr,
            "text": text[ts:te],
        })
    return segments
