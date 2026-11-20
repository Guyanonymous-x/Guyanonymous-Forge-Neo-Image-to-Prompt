# Guyanonymous' Image to Prompt: the three jobs (write a prompt, rank wordings, find missing words), image lists,
# .txt saving and the prompt stored in a Forge / A1111 PNG. No Forge imports, so cli.py can run it on its own.

import gc
import os
import threading

import torch
from PIL import Image

from .qwen35_vl import Qwen35VL

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "qwen35_tokenizer.json")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jxl")

DEFAULT_WRITE = (
    "Write a text-to-image prompt that would recreate this picture as closely as possible. Describe, in plain and "
    "specific sentences: the main subject (age, build, skin, face, hair, expression), clothing, pose and what each hand "
    "is doing, any other people or objects and where they are, the setting, the lighting, the camera angle and framing, "
    "and the style or medium. Describe only what is visible. Output only the prompt, as one paragraph."
)

# Prompt length: name -> target words (None = use the custom number)
LENGTHS = {"Short (about 30 words)": 30, "Medium (about 75 words)": 75, "Long (about 150 words)": 150,
           "Very long (about 250 words)": 250, "Custom": None}
DEPTHS = {
    "Main things only": "Cover only the main subject, what they are doing, the setting and the style; leave out small "
                        "details.",
    "Normal": "Cover the main subject, clothing, pose, other people or objects, setting, lighting, camera angle and "
              "style.",
    "Every small detail": "Cover everything visible: the subject's face, skin, hair, body and expression; every piece "
                          "of clothing with colour, material and fit; the exact pose and what each hand is doing; every "
                          "other person and object and where it is; the background; the light source, its direction "
                          "and colour; the camera angle, distance and lens; and the style or medium.",
}


def write_instruction(length, custom_words, depth, extra):
    """The request sent with the picture when writing a prompt: the length and depth wording plus any extra notes."""
    words = LENGTHS.get(length) or int(custom_words)
    parts = ["Write a text-to-image prompt that would recreate this picture as closely as possible.",
             DEPTHS.get(depth, DEPTHS["Normal"]),
             "Describe only what is visible, in plain and specific sentences.",
             f"Use about {words} words.",
             "Output only the prompt, as one paragraph."]
    if (extra or "").strip():
        parts.append(extra.strip())
    return " ".join(parts), words


def tokens_for_words(words, thinking=False):
    """Room for the answer (about 1.4 tokens a word, plus slack); thinking needs room of its own."""
    return int(words * 1.6) + 64 + (1500 if thinking else 0)


DEFAULT_RANK_QUESTION = "Describe this picture."
DEFAULT_MISSING = (
    "This is the prompt that was used to try to make this picture:\n\n\"{prompt}\"\n\n"
    "Compare the picture with the prompt. List (1) what is clearly visible in the picture but missing from the prompt, "
    "and (2) what the prompt asks for that the picture does not show or shows differently. For every item give the "
    "exact words you would add or change in the prompt. Keep it to a short numbered list."
)

_lock = threading.Lock()
_model = None
_tokenizer = None
_stop = threading.Event()


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    return _tokenizer


def loaded_path():
    return _model.path if _model is not None else None


def load(path, progress=None, before_load=None):
    global _model
    if _model is not None and _model.path == path:
        return _model
    unload()
    if before_load:
        before_load()
    _model = Qwen35VL(path, progress=progress)
    return _model


def unload():
    global _model
    if _model is not None:
        _model.unload()
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def request_stop():
    _stop.set()


def stopped():
    return _stop.is_set()


# ---------------------------------------------------------------- images

def list_images(files=None, folder=None, recursive=False):
    out = []
    for f in files or []:
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if p and p.lower().endswith(IMAGE_EXT):
            out.append(p)
    folder = (folder or "").strip().strip('"')
    if folder:
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        if recursive:
            for root, _, names in os.walk(folder):
                out += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(IMAGE_EXT)]
        else:
            out += [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.lower().endswith(IMAGE_EXT)]
    return out


def open_image(path):
    if isinstance(path, Image.Image):  # already a picture (from an image box)
        return path
    img = Image.open(path)
    img.load()
    return img


def prompt_from_png(path):
    """The positive prompt Forge / A1111 saved in the image (text before 'Negative prompt:' / 'Steps:'), or ''."""
    def read(img):
        text = img.info.get("parameters") or ""
        if not text and hasattr(img, "getexif"):
            ex = img.getexif().get_ifd(0x8769).get(0x9286)  # EXIF UserComment (JPEG/WebP)
            if isinstance(ex, bytes):
                text = ex[8:].decode("utf-16-be" if ex[:8] == b"UNICODE\x00" else "utf-8", "ignore")
            elif isinstance(ex, str):
                text = ex
        return text

    try:
        if isinstance(path, Image.Image):
            text = read(path)
        else:
            with Image.open(path) as img:
                text = read(img)
    except Exception:
        return ""
    for cut in ("\nNegative prompt:", "\nSteps:"):
        if cut in text:
            text = text.split(cut, 1)[0]
    return text.strip()


def txt_path(image_path, suffix=""):
    return os.path.splitext(image_path)[0] + suffix + ".txt"


def save_txt(image_path, text, suffix="", overwrite=False):
    p = txt_path(image_path, suffix)
    if os.path.exists(p) and not overwrite:
        return None
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.strip() + "\n")
    return p


def read_wordings(text, file_path=None):
    lines = []
    for src in (text or "").splitlines():
        lines.append(src)
    fp = (file_path or "").strip().strip('"')
    if fp:
        with open(fp, encoding="utf-8", errors="replace") as f:
            lines += f.read().splitlines()
    seen, out = set(), []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------- jobs

def write_prompt(model, image_path, instruction, **gen):
    return model.generate(tokenizer(), open_image(image_path), instruction or DEFAULT_WRITE, should_stop=stopped, **gen)


def missing_words(model, image_path, prompt, template, **gen):
    text = (template or DEFAULT_MISSING).replace("{prompt}", prompt.strip())
    return model.generate(tokenizer(), open_image(image_path), text, should_stop=stopped, **gen)


def rank(model, image_path, question, wordings, max_tokens=1024):
    """Each wording scored with the picture and again without it. The 'picture lift' (average log-probability per
    piece with the picture minus without) is how much seeing the picture makes the model more likely to say it, so
    ordinary-sounding phrases do not win just for being ordinary. Sorted by lift, highest first.
    Returns dicts: wording, avg (with picture), total, pieces, avg_blind (without picture), lift."""
    q = question or DEFAULT_RANK_QUESTION
    seen = model.score(tokenizer(), open_image(image_path), q, wordings, max_tokens=max_tokens, should_stop=stopped)
    blind = {w: m for w, m, _, _ in model.score(tokenizer(), None, q, wordings, should_stop=stopped)}
    out = []
    for w, mean, total, n in seen:
        b = blind.get(w, float("nan"))
        out.append({"wording": w, "avg": mean, "total": total, "pieces": n, "avg_blind": b, "lift": mean - b})
    out.sort(key=lambda r: -r["lift"] if r["lift"] == r["lift"] else float("inf"))
    return out
