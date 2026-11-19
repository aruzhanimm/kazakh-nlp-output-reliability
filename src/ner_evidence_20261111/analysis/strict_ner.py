"""Dependency-free exact NER scoring and proposed window merge.

This isolated evidence utility does not modify the training/inference pipeline.
Gold is strict IOB2. Prediction repair, when requested, is explicit and counted.
"""


def bio_to_spans(tags, repair_prediction=False):
    spans = []
    active = None
    begin = None
    repairs = 0
    for i, tag in enumerate([*tags, "O"]):
        if tag == "O":
            prefix, kind = "O", None
        elif isinstance(tag, str) and len(tag) > 2 and tag[:2] in ("B-", "I-"):
            prefix, kind = tag[0], tag[2:]
        else:
            raise ValueError("Invalid BIO tag")
        if prefix == "I" and active != kind:
            if not repair_prediction:
                raise ValueError("Illegal IOB2 transition")
            prefix = "B"
            repairs += 1
        if active is not None and (prefix != "I" or kind != active):
            spans.append((begin, i, active))
            active, begin = None, None
        if prefix == "B":
            begin, active = i, kind
    return spans, repairs


def exact_counts(gold, predicted):
    gold, predicted = set(map(tuple, gold)), set(map(tuple, predicted))
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def prf(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def failure_aware_exact(gold, predicted, failed):
    return not failed and set(map(tuple, gold)) == set(map(tuple, predicted))


def merge_window_spans(windows, local_predictions, n_words):
    """Proposed v2 ownership: maximum whole-span edge margin, then earliest window.

    None means an explicitly terminal failed window, never a missing record.
    Inputs must already have been aligned without gold by the ordinary parser.
    Every original word must be covered. Gold is deliberately not an argument.
    """
    if len(windows) != len(local_predictions) or not windows or n_words < 1:
        raise ValueError("Incomplete window records")
    if any(not (0 <= a < b <= n_words) for a, b in windows):
        raise ValueError("Invalid window")
    if any(not any(a <= i < b for a, b in windows) for i in range(n_words)):
        raise ValueError("Uncovered original words")
    accepted, rejected = [], []
    failed_windows = [i for i, p in enumerate(local_predictions) if p is None]
    for wi, predictions in enumerate(local_predictions):
        if predictions is None:
            continue
        lo, hi = windows[wi]
        for ci, (a, b, kind) in enumerate(predictions):
            if not (0 <= a < b <= hi - lo):
                raise ValueError("Unaligned local span")
            start, end = lo + a, lo + b
            owners = [(min(start - x, y - end), -j, j)
                      for j, (x, y) in enumerate(windows) if x <= start and end <= y]
            owner = max(owners)[2]
            if owner != wi:
                rejected.append({"window": wi, "candidate": ci, "reason": "nonowner"})
                continue
            span = (start, end, kind)
            if span in accepted:
                rejected.append({"window": wi, "candidate": ci, "reason": "duplicate"})
            elif any(start < y and x < end for x, y, _ in accepted):
                rejected.append({"window": wi, "candidate": ci, "reason": "overlap_first_valid_wins"})
            else:
                accepted.append(span)
    return {"spans": accepted, "failed_windows": failed_windows,
            "failed": bool(failed_windows), "rejected": rejected}
