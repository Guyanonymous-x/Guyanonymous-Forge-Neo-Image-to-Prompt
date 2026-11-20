# Changelog

## v1.0.0 (2026-11-20)

First release.

- **Image to Prompt** tab in Forge Neo, using the Qwen 3.5 9B text encoder of MOSS-Image (with its own vision part)
  to read pictures.
- **Write a prompt** for pictures or folders: length (short to very long, or custom words), how much detail, extra
  instructions, a view of the full request; Send to txt2img / img2img; save to a file (replace or add to the end);
  optional .txt beside each image. Prompts saved inside image files are ignored.
- **Rank my wordings** by picture lift (score with the picture minus score without it), from typed lines or a
  wildcard file.
- **Find missing words**: compares a picture with your prompt or the prompt saved in it.
- Load button with a warning before Forge's models are moved off the card; unloads itself before any image
  generation.
- API routes under `/guy_i2p/` (status, load, unload, stop, write, rank, missing).
- `cli.py` for use without Forge.
