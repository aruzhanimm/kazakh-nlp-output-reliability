"""Shared visible text. No tokenizer truncation and no silent word loss."""
import re
from .common import ROOT, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .models import load_tokenizer


def align_first_subwords(word_ids, labels, label2id):
    seen, aligned = set(), []
    for wid in word_ids:
        if wid is None or wid in seen:
            aligned.append(-100)
        else:
            if not 0 <= wid < len(labels):
                raise ValueError("Bad tokenizer word index")
            aligned.append(label2id[labels[wid]])
            seen.add(wid)
    if seen != set(range(len(labels))):
        raise ValueError("Tokenizer omitted one or more words")
    return aligned


def first_word_predictions(word_ids, predictions, n_words):
    result = [None] * n_words
    if len(word_ids) != len(predictions):
        raise ValueError("Subword length mismatch")
    for wid, pred in zip(word_ids, predictions):
        if wid is not None and result[wid] is None:
            result[wid] = pred
    if any(x is None for x in result):
        raise ValueError("Incomplete word predictions")
    return result


def windows(tokens, tokenizer, max_tokens=384, max_words=96, overlap=24):
    result, start = [], 0
    while start < len(tokens):
        end = min(len(tokens), start + max_words)
        while end > start and len(tokenizer(tokens[start:end], is_split_into_words=True)["input_ids"]) > max_tokens:
            end -= 1
        if end == start:
            raise ValueError("A single original word exceeds encoder context; revise shared policy")
        encoded = tokenizer(tokens[start:end], is_split_into_words=True)
        if set(x for x in encoded.word_ids() if x is not None) != set(range(end - start)):
            raise ValueError("A source word has no subword representation")
        result.append({"start": start, "end": end, "tokens": tokens[start:end], "text": " ".join(tokens[start:end])})
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
    return result


def owners(windows_, n):
    result = []
    for word in range(n):
        candidates = [(min(word - w["start"], w["end"] - 1 - word), -i, i)
                      for i, w in enumerate(windows_) if w["start"] <= word < w["end"]]
        if not candidates:
            raise ValueError("Window coverage gap")
        result.append(max(candidates)[2])
    return result


def merge_window_tags(windows_, tags, n):
    if len(windows_) != len(tags) or any(len(t) != w["end"] - w["start"] for w, t in zip(windows_, tags)):
        raise ValueError("Incomplete window predictions")
    owner = owners(windows_, n)
    return [tags[owner[i]][i - windows_[owner[i]]["start"]] for i in range(n)]


def sa_prefix(text, tokenizer, max_tokens=384, max_words=128):
    matches = list(re.finditer(r"\S+", text))
    count = min(len(matches), max_words)
    while count:
        visible = text if count == len(matches) else text[:matches[count - 1].end()]
        if len(tokenizer(visible)["input_ids"]) <= max_tokens:
            return visible
        count -= 1
    if matches:
        # Unbroken repeated-character reviews exist. Preserve a nonempty literal
        # prefix, shared by every system, instead of excluding or silently losing it.
        end = matches[0].end()
        while end > 0:
            visible = text[:end]
            if len(tokenizer(visible)["input_ids"]) <= max_tokens:
                return visible
            end -= 1
        raise ValueError("No character prefix fits the shared context")
    return text


def prepare_inputs():
    tok = load_tokenizer("encoder")
    source = ROOT / "data/derived"
    dest = ROOT / "data/pilot"
    policy = {"encoder_max_tokens": 384, "ner_max_words": 96, "ner_overlap_words": 24,
              "ownership": "maximum distance to window edge; earliest window breaks ties",
              "sa_max_words": 128, "sa_policy": "longest whole-word original prefix within both caps; if first word alone exceeds budget, longest character prefix that fits",
              "source_manifest": file_hash(source / "manifest.json"),
              "model_lock": read_json(ROOT / "configs/models.lock.json"), "files": {}}
    for task in ("ner", "sa"):
        policy[task] = {}
        for split in ("train", "valid", "demos"):
            name = f"{task}_demos.jsonl" if split == "demos" else f"{task}_pilot_{split}.jsonl"
            rows = read_jsonl(source / name)
            for row in rows:
                if task == "ner":
                    row["windows"] = windows(row["tokens"], tok)
                    row["owners"] = owners(row["windows"], len(row["tokens"]))
                else:
                    row["visible_text"] = sa_prefix(row["text"], tok)
                    row["capped"] = row["visible_text"] != row["text"]
                    row["character_fallback"] = row["capped"] and len(row["visible_text"]) < len(row["text"].split()[0]) if row["text"].split() else False
            output = dest / f"{task}_{split}.jsonl"
            write_jsonl(output, rows)
            policy["files"][output.name] = file_hash(output)
            lengths = sorted(len(r["tokens"]) if task == "ner" else len(r["text"].split()) for r in rows)
            policy[task][split] = {"rows": len(rows), "original_word_max": max(lengths),
                                  "original_word_p95": lengths[int(.95 * (len(lengths) - 1))],
                                  "windowed_or_capped": sum(len(r["windows"]) > 1 if task == "ner" else r["capped"] for r in rows)}
    write_json(dest / "manifest.json", policy)
    print({t: policy[t] for t in ("ner", "sa")})


def load_panel(task, split):
    if task not in ("ner", "sa") or split not in ("train", "valid", "demos"):
        raise ValueError("Development-only loader refuses test data")
    manifest = read_json(ROOT / "data/pilot/manifest.json")
    if manifest["model_lock"] != read_json(ROOT / "configs/models.lock.json"):
        raise ValueError("Tokenizer/model lock changed; rebuild shared inputs")
    path = ROOT / "data/pilot" / f"{task}_{split}.jsonl"
    if file_hash(path) != manifest["files"][path.name]:
        raise ValueError("Shared input checksum mismatch")
    rows = read_jsonl(path)
    expected = "train" if split in ("train", "demos") else "valid"
    if any(not r["id"].startswith(f"{task}:{expected}:") for r in rows):
        raise ValueError("Wrong source split")
    return rows


if __name__ == "__main__":
    prepare_inputs()
