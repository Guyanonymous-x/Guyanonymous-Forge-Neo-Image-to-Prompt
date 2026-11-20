# Image to Prompt (Forge Neo extension)

Turn pictures back into prompts, using **the same Qwen 3.5 model that MOSS-Image reads your prompts with**.

A text encoder only turns words into numbers, so it can't simply be run backwards. But the Qwen 3.5 9B encoder used by
MOSS-Image is a full vision-language model: its picture-reading part and its text-writing part are still inside the
same file. This extension loads that file and uses it to:

- **Write a prompt** for a picture, a batch of pictures or a whole folder, in the model's own words.
- **Rank my wordings**: score several ways of saying the same thing against a picture, to see which one the model
  links most strongly to what it sees.
- **Find missing words**: compare a picture with the prompt that made it, and list what is missing or came out
  differently, with the exact words to add or change.

It is meant for the prompts that keep failing: if the encoder that reads your prompt describes a picture in different
words than yours, its words are worth trying.

Everything also works through **Forge's API**, so scripts can use it.

## Requirements

- **Forge Neo** (it uses Forge Neo's model folders and memory handling).
- An **NVIDIA graphics card with about 13 GB of free memory** (tested on a 16 GB RTX 5060 Ti).
- A **Qwen 3.5 9B safetensors file that still has its vision part**, in `models/text_encoder`. The INT8 file
  `qwen3.5_9b_int8_convrot.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image)
  (the one the MOSS-Image extension uses) works. bf16/fp16 files work too: they are turned into INT8 while loading so
  they fit the card. Files without the vision part are refused with a message.

The MOSS-Image extension itself is not needed, though the two work well together.

## Install

**From URL (recommended):** in Forge Neo open **Extensions → Install from URL**, paste

```
https://github.com/Guyanonymous-x/Guyanonymous-Forge-Neo-Image-to-Prompt
```

click **Install**, then restart Forge Neo.

**Manually:** download the zip from the
[latest release](https://github.com/Guyanonymous-x/Guyanonymous-Forge-Neo-Image-to-Prompt/releases/latest), extract it
into Forge Neo's `extensions` folder, and restart.

A new **Image to Prompt** tab appears.

## Use

1. Open the **Image to Prompt** tab and press **Load Qwen 3.5 (frees the card)**.
   Loading moves Forge's own image model off the graphics card to make room. Forge loads it back by itself the next
   time you generate an image, and generating an image unloads Qwen 3.5 again, so the two never fight over the card.
   The first load takes about a minute; later loads are quicker while the file is still in memory.
2. Pick a sub-tab.

### Write a prompt

- Add pictures, or type a folder (tick **Include subfolders** to go deeper).
- **Prompt length**: Short (about 30 words), Medium (75), Long (150), Very long (250) or Custom (10 to 500).
- **How much detail**: Main things only, Normal, or Every small detail.
- **Extra instructions** (optional), for example "Start with the camera angle." They steer how it writes; leave the
  box empty to see its untouched wording. **See or write the full request** shows exactly what is sent with each picture; tick **Use my own full request** to write the whole request yourself.
- Only the picture itself is read. A prompt saved inside the image file is ignored, so it can't steer the result.
- Results show with a word count. Pick one under **Prompt from picture**, edit it if you like, then
  **Send to txt2img**, **Send to img2img** (fills the prompt box only; your negative prompt and settings stay), or
  **Save prompt to file** (replace the file, or add to its end; "As one line" keeps wildcard files tidy).
- For folders, **Save a .txt beside each image** writes the prompt next to every picture.

### Rank my wordings

- Add one picture and type several wordings, one per line, or give a wildcard `.txt` file.
- Each wording is scored twice: with the picture, and without it. **Picture lift** (the ranking) is how much seeing
  the picture raises the chance of the model saying exactly those words. That way an ordinary-sounding phrase doesn't
  win just for being ordinary.
- Lifts are most telling when you compare **the same wording across several pictures**. Within one picture, short
  plain wordings tend to come out on top.

### Find missing words

- Add pictures or a folder, and type the prompt, or leave the box empty to use the prompt Forge saved in each picture.
- It lists what is visible but missing from the prompt, and what the prompt asks for that the picture shows
  differently, with the exact words to change, often followed by a revised prompt.

### Model and settings

- **Image detail**: how many picture tokens it reads (default 1024; more is finer and slower).
- **Temperature / Top P / Repetition penalty / Seed**: how varied the writing is. Temperature 0 always gives the same
  answer.
- **Let it think first**: slower, sometimes more careful; only the answer is kept.
- **Stop** ends a batch after the current picture; **Unload model** frees the card.

### Speed (16 GB RTX 5060 Ti)

| Job | Time |
|---|---|
| Load (first time / file cached) | about 1 minute / about 5 seconds |
| Write, medium length | 10 to 30 seconds a picture |
| Write, long | about 55 seconds a picture |
| Rank, five wordings | about 3 seconds |
| Find missing words | about 60 seconds a picture |

### Good to know

- It describes scenes, settings, light and camera angles well, but it can misjudge fine body details such as pose,
  state or size. Treat its descriptions as a second opinion; **Find missing words** and comparing lifts across
  pictures are the more reliable checks.
- Errors show on the page and are also written to `errors.log` in the extension folder.

## API

Forge Neo must be started with `--api`. Bodies are JSON. Pictures are file paths on the Forge computer, or base64
pictures (a `data:image/png;base64,` start is fine).

Jobs need the model loaded: call `/guy_i2p/load` first, or add `"auto_load": true` to a job (this moves Forge's
models off the card, as the Load button does).

| Route | Body | Returns |
|---|---|---|
| `GET /guy_i2p/status` | | loaded, model, model files, lengths, depths, graphics card use |
| `POST /guy_i2p/load` | `{"model": optional file name}` | loaded file, seconds |
| `POST /guy_i2p/unload` | | |
| `POST /guy_i2p/stop` | | stops a running job after its current step |
| `POST /guy_i2p/write` | `images` and/or `folder`, `recursive`, `length` (short / medium / long / very long / custom), `words`, `depth` (Main things only / Normal / Every small detail), `extra`, `request` (your own full request; replaces length, depth and extra), `max_new` (longest answer in tokens; 0 = from the length), `save`, `suffix`, `overwrite` | the request sent, and per picture: prompt, words, saved .txt |
| `POST /guy_i2p/rank` | `image`, `wordings` (list), `wordings_file`, `question` | per wording: rank, picture_lift, fit_percent with and without the picture, log-probabilities, pieces |
| `POST /guy_i2p/missing` | `images` and/or `folder`, `prompt`, `use_png`, `template` (`{prompt}` is replaced), `max_new` | per picture: prompt used and where it came from, answer |

Every job also takes `model`, `auto_load`, `detail` (default 1024), `temperature` (0.7), `top_p` (0.8),
`repetition_penalty` (1.05), `seed` (-1 = random) and `thinking` (false).

Examples:

```bash
curl -X POST http://127.0.0.1:7860/guy_i2p/load -H "Content-Type: application/json" -d "{}"
```

```bash
curl -X POST http://127.0.0.1:7860/guy_i2p/rank -H "Content-Type: application/json" -d "{\"image\": \"C:/pictures/test.png\", \"wordings\": [\"a worm's-eye view through tall grass\", \"an aerial view of a field\"]}"
```

```python
import json, urllib.request
body = {"images": ["C:/pictures/test.png"], "length": "long", "depth": "Every small detail"}
req = urllib.request.Request("http://127.0.0.1:7860/guy_i2p/write", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=600))["results"][0]["prompt"])
```

## Without Forge

`cli.py` runs the same jobs from a command line with Forge Neo's Python, while Forge is closed (so the card is free):

```bash
venv\Scripts\python.exe extensions\Guyanonymous-Forge-Neo-Image-to-Prompt\cli.py --model models\text_encoder\qwen3.5_9b_int8_convrot.safetensors write C:\pictures
```

`rank <image> --wordings "one" "two"` and `missing <images> --prompt "..."` work the same way.

## How it works

Plain PyTorch, no extra packages: Forge Neo's transformers version has no Qwen 3.5 yet. The extension builds the
Qwen 3.5 9B vision tower and language tower itself (Gated DeltaNet and gated attention layers, interleaved MRoPE
positions for picture tokens), adds the cache needed to write text one token at a time, and reads the weights straight
from the safetensors file, keeping INT8 weights as INT8 and the token table in system memory. Ranking compares the
model's log-probability of each wording after the picture with the same wording after no picture.

## Credits

- **[MOSS-Image for Forge Neo](https://github.com/eduardoabreu81/moss-image-forge-neo)** by eduardoabreu81: the
  Qwen 3.5 language tower port this extension's text model follows.
- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** and its Qwen 3.5 vision tower.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous and Comfy-Org: the Qwen 3.5
  implementation and the INT8 files.
- **[Transformers](https://github.com/huggingface/transformers)** by Hugging Face: the Qwen 3.5 and Qwen3-VL reference
  maths.
- **[Qwen 3.5](https://huggingface.co/Qwen)** by the Qwen team: the model and its tokenizer (included as
  `guy_i2p/qwen35_tokenizer.json`, Apache 2.0).
- **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)** by OpenMOSS.

## License

GNU Affero General Public License v3.0 (see `LICENSE`), the same as Forge Neo and MOSS-Image for Forge Neo, whose code
this builds on. The bundled Qwen 3.5 tokenizer is Apache 2.0 (see `NOTICE`).

Made by Guyanonymous-x.
