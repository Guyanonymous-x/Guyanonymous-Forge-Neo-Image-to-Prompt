# Image to Prompt, without Forge (for testing, or when Forge is closed). Run with Forge's Python:
#   venv\Scripts\python.exe extensions\Guyanonymous-Forge-Neo-Image-to-Prompt\cli.py --model <file> write <images or folder>
#   ... rank <image> --wordings "one" "two"          ... missing <image> [--prompt "..."]

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guy_i2p import jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--detail", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=450)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--save", action="store_true")
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("write")
    w.add_argument("paths", nargs="+")
    r = sub.add_parser("rank")
    r.add_argument("image")
    r.add_argument("--question", default=jobs.DEFAULT_RANK_QUESTION)
    r.add_argument("--wordings", nargs="*", default=[])
    r.add_argument("--file")
    m = sub.add_parser("missing")
    m.add_argument("paths", nargs="+")
    m.add_argument("--prompt", default="")
    a = ap.parse_args()

    t0 = time.time()
    model = jobs.load(a.model, progress=lambda f, d: print(f"\r{d}", end="", flush=True))
    print(f"\nloaded in {time.time() - t0:.0f} s")
    gen = dict(max_new=a.max_new, temperature=a.temperature, seed=a.seed, max_tokens=a.detail)

    def images(paths):
        out = []
        for p in paths:
            out += jobs.list_images(folder=p) if os.path.isdir(p) else [p]
        return out

    if a.mode == "write":
        for p in images(a.paths):
            t = time.time()
            text = jobs.write_prompt(model, p, None, **gen)
            print(f"\n### {p} ({time.time() - t:.0f} s)\n{text}")
            if a.save:
                print("saved", jobs.save_txt(p, text))
    elif a.mode == "missing":
        for p in images(a.paths):
            prompt = a.prompt or jobs.prompt_from_png(p)
            t = time.time()
            print(f"\n### {p} ({time.time() - t:.0f} s)\nprompt: {prompt}\n{jobs.missing_words(model, p, prompt, None, **gen)}")
    else:
        t = time.time()
        res = jobs.rank(model, a.image, a.question, jobs.read_wordings("\n".join(a.wordings), a.file), a.detail)
        print(f"({time.time() - t:.0f} s)")
        for i, r in enumerate(res, 1):
            print(f"{i:3d}  lift {r['lift']:+6.2f}  with {r['avg']:7.3f}  without {r['avg_blind']:7.3f}  "
                  f"{r['pieces']:3d}  {r['wording']}")
    jobs.unload()


if __name__ == "__main__":
    main()
