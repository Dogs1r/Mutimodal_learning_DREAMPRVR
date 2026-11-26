from hashlib import sha256
from pathlib import Path


def development_captions(source, run_dir):
    lines = Path(source).read_text(encoding="utf-8").splitlines(keepends=True)
    ids = [line.split("#", 1)[0] for line in lines]
    ordered = sorted(set(ids), key=lambda vid: sha256(("AdaptivePRVR-v1/" + vid).encode()).digest())
    if len(ordered) != 5338:
        raise ValueError("The Charades development split expects 5338 training videos")
    development = set(ordered[:534])
    output = {}
    for name in ("train", "val"):
        path = Path(run_dir) / (name + ".caption.txt")
        contents = "".join(line for vid, line in zip(ids, lines)
                           if (vid in development) == (name == "val"))
        if path.exists():
            if path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"Split differs from existing manifest: {path}")
        else:
            path.write_text(contents, encoding="utf-8")
        output[name] = str(path)
    return output
