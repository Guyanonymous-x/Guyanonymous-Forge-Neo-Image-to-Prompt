# Guyanonymous' Image to Prompt: a Forge Neo tab that uses the Qwen3.5 text encoder MOSS-Image prompts with (its own
# vision tower included in the file) to describe pictures, rank wordings against a picture, and find what a prompt
# misses. The model is loaded only when a job runs and is unloaded before any image generation starts.

import functools
import importlib
import os
import sys
import time
import traceback

import gradio as gr

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)

# 'Reload UI' runs this file again: free the old model and pick up changed code
if "guy_i2p.jobs" in sys.modules:
    try:
        sys.modules["guy_i2p.jobs"].unload()
    except Exception:
        pass
    importlib.reload(sys.modules["guy_i2p.qwen35_vl"])
    importlib.reload(sys.modules["guy_i2p.jobs"])

from guy_i2p import jobs  # noqa: E402
from modules import paths, script_callbacks, scripts  # noqa: E402

SAVE_DEFAULT = os.path.join(paths.data_path, "output", "image_to_prompt", "prompts.txt")


def show_errors(fn):
    """Any failure shows its message on the page (and the full details in Forge's console)."""
    def report(e):
        if isinstance(e, gr.Error):
            raise e
        traceback.print_exc()
        try:
            with open(os.path.join(EXT_DIR, "errors.log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("\n--- %Y-%m-%d %H:%M:%S\n") + traceback.format_exc())
        except OSError:
            pass
        raise gr.Error(f"{type(e).__name__}: {e}", duration=None)

    if getattr(fn, "__code__", None) and fn.__code__.co_flags & 0x20:  # generator
        @functools.wraps(fn)
        def gen(*a, **k):
            try:
                yield from fn(*a, **k)
            except Exception as e:
                report(e)
        return gen

    @functools.wraps(fn)
    def plain(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            report(e)
    return plain

TE_DIR = os.path.join(paths.models_path, "text_encoder")
# tried first, in this order: Comfy-Org's INT8 file for MOSS-Image, then any other INT8 Qwen 3.5
PREFERRED = ("qwen3.5_9b_int8_convrot.safetensors", "qwen35ForMossImage_v10_int8.safetensors")


def model_choices():
    out = []
    if os.path.isdir(TE_DIR):
        for root, _, names in os.walk(TE_DIR):
            for n in names:
                low = n.lower()
                if low.endswith(".safetensors") and any(k in low for k in ("qwen35", "qwen3.5", "qwen3_5")):
                    out.append(os.path.relpath(os.path.join(root, n), TE_DIR))
    return sorted(out)


def default_model(choices):
    for c in choices:
        if os.path.basename(c) in PREFERRED:
            return c
    for c in choices:
        if "int8" in c.lower():
            return c
    return choices[0] if choices else None


def free_forge_memory():
    try:
        from backend import memory_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(True)
    except Exception as e:  # never stop the job over this
        print(f"[Image to Prompt] could not ask Forge to free the graphics card: {e}")


NOT_LOADED = ("Qwen 3.5 is not loaded yet. Click 'Load Qwen 3.5' at the top of this tab first. "
              "That moves Forge's image model off the graphics card to make room.")


def get_model(choice, progress):
    """The loaded model, if it is the file chosen; never loads by itself (loading moves Forge's models off the card)."""
    if not choice:
        raise gr.Error("No Qwen 3.5 file found in models/text_encoder.")
    if jobs.loaded_path() != os.path.join(TE_DIR, choice):
        raise gr.Error(NOT_LOADED if not jobs.loaded_path() else
                       "A different Qwen 3.5 file is loaded than the one chosen. Click 'Load Qwen 3.5' to switch.")
    return jobs._model


def card_text():
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return f"Graphics card: {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GB in use."
    except Exception:
        return ""


def status_text():
    p = jobs.loaded_path()
    s = f"Loaded: {os.path.basename(p)}." if p else "Qwen 3.5 not loaded."
    return f"{s} {card_text()}"


def do_load(choice, progress=gr.Progress()):
    if not choice:
        raise gr.Error("No Qwen 3.5 file found in models/text_encoder.")
    with jobs._lock:
        progress(0, desc="Moving Forge's models off the graphics card")
        jobs.load(os.path.join(TE_DIR, choice), progress=lambda f, d: progress(f, desc=d),
                  before_load=free_forge_memory)
    return status_text()


def gen_args(max_new, temperature, top_p, rep, seed, detail, thinking):
    return dict(max_new=int(max_new), temperature=float(temperature), top_p=float(top_p),
                repetition_penalty=float(rep), seed=int(seed), max_tokens=int(detail), thinking=bool(thinking))


def _file_paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


# ---------------------------------------------------------------- job runners (generators, so results show as they come)

def run_write(model_choice, detail, max_new, temperature, top_p, rep, seed, thinking,
              files, folder, recursive, length, custom_words, depth, extra, use_own, own_request, save, suffix,
              overwrite, progress=gr.Progress()):
    images = jobs.list_images(_file_paths(files), folder, recursive)
    if not images:
        raise gr.Error("Add images or give a folder.")
    instruction, words = jobs.write_instruction(length, custom_words, depth, extra)
    if use_own:
        if not (own_request or "").strip():
            raise gr.Error("'Use my own full request' is ticked but the request box is empty.")
        instruction = own_request.strip()
    args = gen_args(jobs.tokens_for_words(words, thinking), temperature, top_p, rep, seed, detail, thinking)
    jobs._stop.clear()
    results = {}  # picture name -> prompt, for the send / save box

    def pick():
        names = list(results)
        return (gr.update(choices=names, value=names[-1] if names else None),
                results[names[-1]] if names else "", results)

    with jobs._lock:
        model = get_model(model_choice, progress)
        out, t0 = [], time.time()
        for n, path in enumerate(images, 1):
            if jobs.stopped():
                out.append("Stopped.")
                break
            progress((n - 1) / len(images), desc=f"Image {n} of {len(images)}: {os.path.basename(path)}")
            text = jobs.write_prompt(model, path, instruction, **args)
            results[os.path.basename(path)] = text
            note = f"\n\n({len(text.split())} words)"
            if save and folder and path.startswith(folder.strip().strip('"')):
                saved = jobs.save_txt(path, text, suffix, overwrite)
                note += f"\n(saved {saved})" if saved else "\n(.txt already there, not overwritten)"
            out.append(f"### {os.path.basename(path)}\n{text}{note}")
            yield ("\n\n".join(out), status_text() + f" | {n}/{len(images)} done, {time.time() - t0:.0f} s", *pick())
    yield ("\n\n".join(out), status_text() + f" | finished in {time.time() - t0:.0f} s", *pick())


def save_prompt(text, path, mode, one_line):
    text = (text or "").strip()
    if not text:
        raise gr.Error("There is no prompt to save.")
    if one_line:
        text = " ".join(text.split())
    path = (path or "").strip().strip('"') or SAVE_DEFAULT
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    append = mode.startswith("Add") and os.path.exists(path)
    if append:
        with open(path, "rb") as f:
            f.seek(0, 2)
            needs_newline = False
            if f.tell() > 0:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(("\n" if needs_newline else "") + text + "\n")
        return f"Added to the end of {path}"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return f"Saved to {path}"


def send_js(tab):
    return ("(t) => { const ta = gradioApp().querySelector('#%s_prompt textarea'); if (ta) { ta.value = t; "
            "updateInput(ta); } switch_to_%s(); return []; }") % (tab, tab)


def run_missing(model_choice, detail, max_new, temperature, top_p, rep, seed, thinking,
                files, folder, recursive, prompt, use_png, template, save, suffix, overwrite, progress=gr.Progress()):
    images = jobs.list_images(_file_paths(files), folder, recursive)
    if not images:
        raise gr.Error("Add images or give a folder.")
    jobs._stop.clear()
    with jobs._lock:
        model = get_model(model_choice, progress)
        out, t0 = [], time.time()
        for n, path in enumerate(images, 1):
            if jobs.stopped():
                out.append("Stopped.")
                break
            used = (prompt or "").strip()
            src = "your prompt"
            if not used and use_png:
                used = jobs.prompt_from_png(path)
                src = "the prompt saved in the image"
            if not used:
                out.append(f"### {os.path.basename(path)}\n(no prompt: type one, or use images saved by Forge)")
                continue
            progress((n - 1) / len(images), desc=f"Image {n} of {len(images)}: {os.path.basename(path)}")
            text = jobs.missing_words(model, path, used, template, **gen_args(max_new, temperature, top_p, rep, seed, detail, thinking))
            note = ""
            if save and folder and path.startswith(folder.strip().strip('"')):
                saved = jobs.save_txt(path, text, suffix, overwrite)
                note = f"\n(saved {saved})" if saved else "\n(.txt already there, not overwritten)"
            out.append(f"### {os.path.basename(path)}\nChecked against {src}.\n\n{text}{note}")
            yield "\n\n".join(out), status_text() + f" | {n}/{len(images)} done, {time.time() - t0:.0f} s"
    yield "\n\n".join(out), status_text() + f" | finished in {time.time() - t0:.0f} s"


def run_rank(model_choice, detail, image, question, wordings_text, wordings_file, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Add one image.")
    try:
        wordings = jobs.read_wordings(wordings_text, wordings_file)
    except OSError as e:
        raise gr.Error(f"Could not read the wordings file: {e}")
    if not wordings:
        raise gr.Error("Type at least one wording (one per line).")
    jobs._stop.clear()
    t0 = time.time()
    with jobs._lock:
        model = get_model(model_choice, progress)
        progress(0.5, desc=f"Scoring {len(wordings)} wordings")
        res = jobs.rank(model, image, question, wordings, max_tokens=int(detail))
    rows = ["| Rank | Wording | Picture lift | Fit % with picture | Fit % without | Pieces |",
            "|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(res, 1):
        w = r["wording"].replace("|", "/")
        if r["lift"] != r["lift"]:
            rows.append(f"| {i} | {w} | | | | {r['pieces']} |")
        else:
            rows.append(f"| {i} | {w} | {r['lift']:+.2f} | {100 * 2.718281828 ** r['avg']:.1f} | "
                        f"{100 * 2.718281828 ** r['avg_blind']:.1f} | {r['pieces']} |")
    return "\n".join(rows), status_text() + f" | {len(res)} wordings scored in {time.time() - t0:.0f} s"


def do_unload():
    with jobs._lock:
        jobs.unload()
    return status_text()


def do_stop():
    jobs.request_stop()
    return "Stopping after the current step..."


# ---------------------------------------------------------------- UI

def folder_inputs():
    with gr.Row():
        files = gr.File(label="Images", file_count="multiple", file_types=["image"])
        with gr.Column():
            folder = gr.Textbox(label="Or a folder of images", placeholder=r"D:\pictures\tests")
            recursive = gr.Checkbox(label="Include subfolders", value=False)
    return files, folder, recursive


def save_inputs(default_suffix):
    with gr.Row():
        save = gr.Checkbox(label="Save a .txt beside each image (folder images only)", value=False)
        suffix = gr.Textbox(label="Add to the .txt name", value=default_suffix, scale=0, min_width=140)
        overwrite = gr.Checkbox(label="Overwrite existing .txt", value=False)
    return save, suffix, overwrite


def on_ui_tabs():
    choices = model_choices()
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown(
            "Uses the **same Qwen 3.5 model MOSS Image reads your prompts with** to look at pictures. "
            "It describes them in its own words, scores your wordings against them, and lists what a prompt misses."
        )
        with gr.Row():
            gr.Markdown(
                "⚠️ **Qwen 3.5 needs about 13 GB of the graphics card.** 'Load Qwen 3.5' first moves Forge's image "
                "model (and anything else it has loaded) off the card; Forge loads it back by itself on your next "
                "image. Generating an image unloads Qwen 3.5 again."
            )
            load = gr.Button("Load Qwen 3.5 (frees the card)", variant="primary", scale=0)
        with gr.Accordion("Model and settings", open=False):
            with gr.Row():
                model_choice = gr.Dropdown(label="Qwen 3.5 file (models/text_encoder)", choices=choices,
                                           value=default_model(choices))
                refresh = gr.Button("Refresh list", scale=0)
            with gr.Row():
                detail = gr.Slider(label="Image detail (picture tokens; more = finer detail, slower)",
                                   minimum=256, maximum=4096, step=64, value=1024)
                max_new = gr.Slider(label="Longest answer for 'Find missing words' (tokens)", minimum=32,
                                    maximum=2048, step=16, value=450)
            with gr.Row():
                temperature = gr.Slider(label="Temperature (0 = always the same answer)", minimum=0, maximum=1.5,
                                        step=0.05, value=0.7)
                top_p = gr.Slider(label="Top P", minimum=0.05, maximum=1, step=0.05, value=0.8)
                rep = gr.Slider(label="Repetition penalty", minimum=1, maximum=1.5, step=0.01, value=1.05)
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            thinking = gr.Checkbox(label="Let it think first (slower, sometimes more careful; only the answer is kept)",
                                   value=False)
        with gr.Row():
            status = gr.Textbox(label="Status", value=status_text(), interactive=False, scale=4)
            stop = gr.Button("Stop", scale=0)
            unload = gr.Button("Unload model", scale=0)
        settings = [model_choice, detail, max_new, temperature, top_p, rep, seed, thinking]

        with gr.Tab("Write a prompt"):
            gr.Markdown("Only the picture itself is read. Any prompt or settings saved inside the image file are "
                        "ignored here, so they cannot steer what it writes.")
            w_files, w_folder, w_rec = folder_inputs()
            with gr.Row():
                w_len = gr.Radio(label="Prompt length", choices=list(jobs.LENGTHS), value="Medium (about 75 words)")
                w_words = gr.Slider(label="Custom length (words)", minimum=10, maximum=500, step=5, value=100)
            w_depth = gr.Radio(label="How much detail", choices=list(jobs.DEPTHS), value="Normal")
            w_extra = gr.Textbox(label="Extra instructions (optional)", lines=2,
                                 placeholder="e.g. Describe her outfit first. Name the camera angle in plain words.")
            with gr.Accordion("See or write the full request sent with each picture", open=False):
                w_custom = gr.Checkbox(label="Use my own full request (replaces the length, detail and extra "
                                             "instructions above; the length still sets how long it may write)",
                                       value=False)
                w_preview = gr.Textbox(show_label=False, interactive=True, lines=5)
            w_save, w_suffix, w_over = save_inputs("")
            w_run = gr.Button("Write prompts", variant="primary")
            with gr.Group():
                with gr.Row():
                    w_pick = gr.Dropdown(label="Prompt from picture", choices=[], scale=1)
                    w_results = gr.State({})
                w_prompt = gr.Textbox(label="Prompt (you can edit it before sending or saving)", lines=4)
                with gr.Row():
                    # own ids: Forge gives a plain "Send to txt2img" button the id txt2img_tab, which its paste
                    # wiring then fills with the last generation's prompt
                    w_to_txt = gr.Button("Send to txt2img", elem_id="guy_i2p_send_txt2img")
                    w_to_img = gr.Button("Send to img2img", elem_id="guy_i2p_send_img2img")
                with gr.Row():
                    w_file = gr.Textbox(label="Save to file", value=SAVE_DEFAULT, scale=3)
                    w_mode = gr.Radio(label="If the file exists", choices=["Add to the end of the file", "Replace the file"],
                                      value="Add to the end of the file", scale=2)
                    w_oneline = gr.Checkbox(label="As one line", value=True, scale=1)
                with gr.Row():
                    w_savebtn = gr.Button("Save prompt to file", scale=0)
                    w_saved = gr.Markdown()
            w_out = gr.Markdown()
            w_pick.change(lambda name, res: res.get(name, ""), inputs=[w_pick, w_results], outputs=[w_prompt])
            w_to_txt.click(None, inputs=[w_prompt], js=send_js("txt2img"))
            w_to_img.click(None, inputs=[w_prompt], js=send_js("img2img"))
            w_savebtn.click(show_errors(save_prompt), inputs=[w_prompt, w_file, w_mode, w_oneline], outputs=[w_saved])
            # the box shows the built request; once "use my own" is ticked, what is typed there is kept and used
            preview = lambda l, c, d, e, own: gr.update() if own else jobs.write_instruction(l, c, d, e)[0]
            for c in (w_len, w_words, w_depth, w_extra, w_custom):
                c.change(preview, inputs=[w_len, w_words, w_depth, w_extra, w_custom], outputs=[w_preview])
            tab.load(preview, inputs=[w_len, w_words, w_depth, w_extra, w_custom], outputs=[w_preview])
            w_run.click(show_errors(run_write), inputs=settings + [w_files, w_folder, w_rec, w_len, w_words, w_depth,
                                                                   w_extra, w_custom, w_preview, w_save, w_suffix,
                                                                   w_over],
                        outputs=[w_out, status, w_pick, w_prompt, w_results])

        with gr.Tab("Rank my wordings"):
            gr.Markdown(
                "Type several wordings for the same thing, one per line (or load a wildcard .txt). Each one is scored by "
                "how likely the model is to say exactly those words, once with the picture and once without it. "
                "**Picture lift** (the ranking) is how much seeing the picture raises that chance: higher means the "
                "model links that wording to what it sees, not just to what sounds ordinary. **Fit %** is the average "
                "chance per word piece with and without the picture."
            )
            with gr.Row():
                r_image = gr.Image(label="Image", type="pil")  # "filepath" breaks under Forge Neo (save_pil_to_cache)
                with gr.Column():
                    r_question = gr.Textbox(label="Question it answers", value=jobs.DEFAULT_RANK_QUESTION)
                    r_words = gr.Textbox(label="Wordings, one per line", lines=8,
                                         placeholder="a woman with a claw-clip updo\na woman with her hair clipped up")
                    r_file = gr.Textbox(label="Or a wordings / wildcard .txt file", placeholder=r"C:\wildcards\hair_styles.txt")
            r_run = gr.Button("Rank wordings", variant="primary")
            r_out = gr.Markdown()
            r_run.click(show_errors(run_rank), inputs=[model_choice, detail, r_image, r_question, r_words, r_file],
                        outputs=[r_out, status])

        with gr.Tab("Find missing words"):
            m_files, m_folder, m_rec = folder_inputs()
            gr.Markdown("This job is the only one that reads a prompt. It uses the prompt you type, or (if you tick "
                        "the box and leave it empty) the prompt Forge saved inside each image.")
            m_prompt = gr.Textbox(label="Your prompt", lines=4)
            m_png = gr.Checkbox(label="When the box is empty, use the prompt saved in the image", value=True)
            m_template = gr.Textbox(label="What to ask it ({prompt} is replaced by the prompt)",
                                    value=jobs.DEFAULT_MISSING, lines=5)
            m_save, m_suffix, m_over = save_inputs("_missing")
            m_run = gr.Button("Find missing words", variant="primary")
            m_out = gr.Markdown()
            m_run.click(show_errors(run_missing), inputs=settings + [m_files, m_folder, m_rec, m_prompt, m_png, m_template, m_save,
                                                        m_suffix, m_over], outputs=[m_out, status])

        refresh.click(lambda: gr.update(choices=model_choices()), outputs=[model_choice])
        load.click(show_errors(do_load), inputs=[model_choice], outputs=[status])
        unload.click(do_unload, outputs=[status])
        stop.click(do_stop, outputs=[status])
    return [(tab, "Image to Prompt", "guy_image_to_prompt")]


script_callbacks.on_ui_tabs(on_ui_tabs)


# ---------------------------------------------------------------- API (needs Forge started with --api)
# GET  /guy_i2p/status   POST /guy_i2p/load   /guy_i2p/unload   /guy_i2p/stop
# POST /guy_i2p/write    /guy_i2p/rank   /guy_i2p/missing
# Pictures are file paths on this computer or base64 pictures (a "data:image/...;base64," start is fine).
# Jobs need Qwen 3.5 loaded: call /load first, or send "auto_load": true (that moves Forge's models off the card).

def on_app_started(demo, app):
    import base64
    import io

    from fastapi import HTTPException
    from PIL import Image
    from pydantic import BaseModel

    class Settings(BaseModel):
        model: str | None = None  # file under models/text_encoder; default: the MOSS INT8 copy
        auto_load: bool = False
        detail: int = 1024  # picture tokens
        temperature: float = 0.7
        top_p: float = 0.8
        repetition_penalty: float = 1.05
        seed: int = -1
        thinking: bool = False

    class Pictures(Settings):
        images: list[str] = []
        folder: str = ""
        recursive: bool = False
        save: bool = False  # .txt beside folder pictures
        suffix: str = ""
        overwrite: bool = False

    class WriteReq(Pictures):
        length: str = "medium"  # short / medium / long / very long / custom
        words: int = 100  # used when length is custom
        depth: str = "Normal"  # Main things only / Normal / Every small detail
        extra: str = ""
        request: str = ""  # the whole request sent with each picture; replaces length, depth and extra when set
        max_new: int = 0  # longest answer in tokens; 0 = worked out from the length

    class MissingReq(Pictures):
        prompt: str = ""
        use_png: bool = True
        template: str = ""
        max_new: int = 450

    class RankReq(Settings):
        image: str
        question: str = ""
        wordings: list[str] = []
        wordings_file: str = ""

    class LoadReq(BaseModel):
        model: str | None = None

    def model_path(name):
        choices = model_choices()
        name = name or default_model(choices)
        if not name:
            raise HTTPException(404, "No Qwen 3.5 file found in models/text_encoder.")
        path = os.path.join(TE_DIR, name)
        if not os.path.isfile(path):
            raise HTTPException(404, f"Not found: {name}. Files: {choices}")
        return path

    def ensure(req):
        path = model_path(req.model)
        if jobs.loaded_path() != path:
            if not req.auto_load:
                raise HTTPException(409, "Qwen 3.5 is not loaded (or another file is). Call /guy_i2p/load first, or "
                                         "send auto_load: true. Loading moves Forge's models off the graphics card.")
            jobs.load(path, before_load=free_forge_memory)
        return jobs._model

    def picture(s):
        """(name, path or PIL image) from a path or base64 text."""
        s = (s or "").strip().strip('"')
        if os.path.isfile(s):
            return os.path.basename(s), s
        try:
            data = base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s, validate=False)
            img = Image.open(io.BytesIO(data))
            img.load()
            return None, img
        except Exception:
            raise HTTPException(400, f"Not a file on this computer or a base64 picture: {s[:80]}")

    def pictures(req):
        out = []
        for n, s in enumerate(req.images, 1):
            name, img = picture(s)
            out.append((name or f"picture {n}", img))
        try:
            out += [(os.path.basename(p), p) for p in jobs.list_images(folder=req.folder, recursive=req.recursive)]
        except ValueError as e:
            raise HTTPException(404, str(e))
        if not out:
            raise HTTPException(400, "No pictures: send images and/or folder.")
        return out

    def length_name(s):
        s = (s or "").strip().lower()
        for name in jobs.LENGTHS:
            if name.lower().startswith(s) or name.lower().split(" (")[0] == s:
                return name
        return "Medium (about 75 words)"

    def depth_name(s):
        for name in jobs.DEPTHS:
            if name.lower() == (s or "").strip().lower():
                return name
        return "Normal"

    def maybe_save(req, img, text, default_suffix):
        if req.save and isinstance(img, str) and req.folder and img.startswith(req.folder.strip().strip('"')):
            return jobs.save_txt(img, text, req.suffix or default_suffix, req.overwrite)
        return None

    def run(fn):
        try:
            return fn()
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    @app.get("/guy_i2p/status")
    def api_status():
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            card = {"card_used_gb": round((total - free) / 2**30, 2), "card_total_gb": round(total / 2**30, 2)}
        except Exception:
            card = {}
        p = jobs.loaded_path()
        return {"loaded": bool(p), "model": os.path.basename(p) if p else None, "models": model_choices(),
                "lengths": list(jobs.LENGTHS), "depths": list(jobs.DEPTHS), **card}

    @app.post("/guy_i2p/load")
    def api_load(req: LoadReq):
        def go():
            t0 = time.time()
            with jobs._lock:
                jobs.load(model_path(req.model), before_load=free_forge_memory)
            return {"loaded": os.path.basename(jobs.loaded_path()), "seconds": round(time.time() - t0, 1)}
        return run(go)

    @app.post("/guy_i2p/unload")
    def api_unload():
        with jobs._lock:
            jobs.unload()
        return {"loaded": False}

    @app.post("/guy_i2p/stop")
    def api_stop():
        jobs.request_stop()
        return {"stopping": True}

    @app.post("/guy_i2p/write")
    def api_write(req: WriteReq):
        def go():
            pics = pictures(req)
            instruction, words = jobs.write_instruction(length_name(req.length), req.words, depth_name(req.depth),
                                                        req.extra)
            if req.request.strip():
                instruction = req.request.strip()
            max_new = req.max_new if req.max_new > 0 else jobs.tokens_for_words(words, req.thinking)
            args = gen_args(max_new, req.temperature, req.top_p, req.repetition_penalty, req.seed, req.detail,
                            req.thinking)
            jobs._stop.clear()
            t0, results = time.time(), []
            with jobs._lock:
                model = ensure(req)
                for name, img in pics:
                    if jobs.stopped():
                        break
                    text = jobs.write_prompt(model, img, instruction, **args)
                    results.append({"image": name, "prompt": text, "words": len(text.split()),
                                    "saved": maybe_save(req, img, text, "")})
            return {"instruction": instruction, "results": results, "stopped": jobs.stopped(),
                    "seconds": round(time.time() - t0, 1)}
        return run(go)

    @app.post("/guy_i2p/missing")
    def api_missing(req: MissingReq):
        def go():
            pics = pictures(req)
            args = gen_args(req.max_new, req.temperature, req.top_p, req.repetition_penalty, req.seed, req.detail,
                            req.thinking)
            jobs._stop.clear()
            t0, results = time.time(), []
            with jobs._lock:
                model = ensure(req)
                for name, img in pics:
                    if jobs.stopped():
                        break
                    used, src = req.prompt.strip(), "request"
                    if not used and req.use_png:
                        used, src = jobs.prompt_from_png(img), "saved in the picture"
                    if not used:
                        results.append({"image": name, "error": "no prompt: send one, or use pictures saved by Forge"})
                        continue
                    text = jobs.missing_words(model, img, used, req.template, **args)
                    results.append({"image": name, "prompt_used": used, "prompt_source": src, "answer": text,
                                    "saved": maybe_save(req, img, text, "_missing")})
            return {"results": results, "stopped": jobs.stopped(), "seconds": round(time.time() - t0, 1)}
        return run(go)

    @app.post("/guy_i2p/rank")
    def api_rank(req: RankReq):
        def go():
            _, img = picture(req.image)
            try:
                wordings = jobs.read_wordings("\n".join(req.wordings), req.wordings_file)
            except OSError as e:
                raise HTTPException(404, f"Could not read the wordings file: {e}")
            if not wordings:
                raise HTTPException(400, "No wordings.")
            jobs._stop.clear()
            t0 = time.time()
            with jobs._lock:
                model = ensure(req)
                res = jobs.rank(model, img, req.question, wordings, max_tokens=req.detail)
            rows = []
            for i, r in enumerate(res, 1):
                ok = r["lift"] == r["lift"]
                rows.append({"rank": i, "wording": r["wording"], "picture_lift": round(r["lift"], 4) if ok else None,
                             "fit_percent": round(100 * 2.718281828 ** r["avg"], 2) if ok else None,
                             "fit_percent_without_picture": round(100 * 2.718281828 ** r["avg_blind"], 2) if ok else None,
                             "avg_logprob": round(r["avg"], 4) if ok else None,
                             "avg_logprob_without_picture": round(r["avg_blind"], 4) if ok else None,
                             "total_logprob": round(r["total"], 3) if ok else None, "pieces": r["pieces"]})
            return {"question": req.question or jobs.DEFAULT_RANK_QUESTION, "results": rows,
                    "seconds": round(time.time() - t0, 1)}
        return run(go)


script_callbacks.on_app_started(on_app_started)


class Script(scripts.Script):
    """Unloads the Qwen 3.5 reader before any txt2img / img2img run, so the two never fight over the card."""

    def title(self):
        return "Image to Prompt (auto unload)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        if jobs.loaded_path():
            with jobs._lock:
                jobs.unload()
            print("[Image to Prompt] unloaded Qwen 3.5 before image generation")
