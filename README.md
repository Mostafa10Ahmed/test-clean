# BearBot — AI Manhwa & Manga Cleaning Bot

Full-stack AI-powered Manhwa & Manga scanlation cleaning bot (Discord Bot + Web Review Studio + Native-Resolution ONNX Cleaning Pipeline) inspired by `k-omiq/Bear`.

## Key Features

- **Explicit SFX Control on `/clean`**:
  - Every `/clean` run lets the user explicitly choose whether to clean Sound Effects (`clean_sfx: true` or `clean_sfx: false`).
  - Discord Slash Command: `/clean clean_sfx:<True|False> [file:<image or .zip>] [chapter:<chapterId>] [bubble_mode:<whiten|text_only>]`
  - Web Console & CLI syntax: `/clean chapter1 sfx:false` or `/clean chapter1 sfx:true mode:text_only`
- **Direct Discord Image & Chapter ZIP Uploads**:
  - Attach a single `.png/.jpg/.webp` page or a full chapter `.zip` directly inside `/clean`, select `clean_sfx`, and receive the cleaned lossless PNG or ZIP right in Discord.
- **4 Independent Cleaning Categories**:
  1. **Speech Bubbles** (Smart Bubble Whitening `#FFFFFF` or Text-Only Removal; colored/textured bubble protection).
  2. **OT (Outside Text)** including tilted banners, red/white ribbons, double-outlined subtitles, and system windows.
  3. **ST (Side Text)** & margin annotations.
  4. **SFX (Stylized Sound Effects)** with manual review flagging.
- **Native-Resolution ONNX Pipeline**:
  - `detector-v4-s_int8.onnx` (RT-DETR Comic Text & Bubble Detector)
  - `comictextdetector.pt.onnx` (CTD pixel-level text segmentation)
  - `lama-manga.onnx` (Manga-LaMa inpainting with native-resolution overlapping sub-crops for wide subtitles)
  - 100% bit-identical pixel preservation outside authorized region masks.

## Quick Start (Local / VPS)

```bash
# 1. Install dependencies
npm install

# 2. Configure environment variables
cp .env.example .env
# Edit .env and set DISCORD_BOT_TOKEN (and optionally GEMINI_API_KEY)

# 3. Pre-download ONNX models (~300 MB total; also auto-downloads on first clean)
npm run download-models

# 4. Run the bot & web studio on port 3000
npm run dev
```

For production builds:

```bash
npm run build
NODE_ENV=production npm start
```

## Docker Deployment

```bash
docker build -t bearbot-cleaner .
docker run -p 3000:3000 --env-file .env -v $(pwd)/models:/app/models bearbot-cleaner
```

## Discord Bot Setup

1. Create an application in the Discord Developer Portal and copy the Bot Token into `DISCORD_BOT_TOKEN` in `.env`.
2. Invite the bot with `bot` and `applications.commands` scopes (Send Messages, Attach Files).
3. Start the server — slash commands (`/clean`, `/detect`, `/review`, `/status`, `/cancel`, `/settings`, `/export`, `/help`) register automatically when the bot connects.

## Pushing Updates to `Mostafa10Ahmed/test-clean`

```bash
git clone https://github.com/Mostafa10Ahmed/test-clean.git
cd test-clean
unzip -o ../test-clean-bot.zip -d .
git add -A
git commit -m "Update BearBot engine, add explicit clean_sfx option in /clean"
git push origin main
```

## License & Attribution

Licensed under `GPL-3.0-or-later` (see `LICENSE`), respecting the upstream `k-omiq/Bear` project and individual ONNX model licenses.
