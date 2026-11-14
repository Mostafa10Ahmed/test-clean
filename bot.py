"""بوت ديسكورد: بياخد فصل (zip) ويرجعلك فصل متبيّض على gofile، بطابور وبدون ما يعطّل باقي الأوامر."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv

from gofile import GofileError, upload_to_gofile
from profiles import PROFILE_CHOICES
from queue_manager import CANCELLED, DONE, FAILED, Job, JobQueue, QueueFullError
from sources import SourceError, download_zip
from cli_cleaner import clean_chapter_cli
from remote_cleaner import clean_chapter_remote
from local_cleaner import clean_chapter_local, ensure_model

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("whiten-bot")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GUILD_ID = _env_int("DISCORD_GUILD_ID", 0)  # 0 = sync global (أبطأ في الظهور)
ADMIN_ROLE_ID = _env_int("ADMIN_ROLE_ID", 0)
ALLOWED_CHANNEL_IDS = {int(x) for x in os.getenv("ALLOWED_CHANNEL_IDS", "").replace(" ", "").split(",") if x}
GOFILE_TOKEN = os.getenv("GOFILE_TOKEN") or None
WORKER_URL = os.getenv("WORKER_URL") or None
WORKER_TOKEN = os.getenv("WORKER_TOKEN") or None
TRANSLATOR_REPO = os.getenv("TRANSLATOR_REPO", "/root/manga-translator-ui")  # فولدر manga-translator-ui
TRANSLATOR_PYTHON = os.getenv("TRANSLATOR_PYTHON", "/root/manga-translator-ui/.venv/bin/python")
WORK_ROOT = os.getenv("WORK_ROOT", "/root/whiten-work")  # فولدر مؤقت لكل فصل
# محرك التبييض:
#   builtin = البوت بيبيّض بنفسه (الافتراضي، مش محتاج أي سيرفر)
#   remote  = يبعت لسيرفر GPU على Colab (WORKER_URL)
#   cli     = يشغّل manga-translator-ui متسطّب على نفس الجهاز
CLEAN_ENGINE = (os.getenv("CLEAN_ENGINE") or ("remote" if WORKER_URL else "builtin")).strip().lower()
MODEL_DIR = os.getenv("MODEL_DIR", "models")
PAGE_CONCURRENCY = _env_int("PAGE_CONCURRENCY", 2)
QUEUE_WORKERS = _env_int("QUEUE_WORKERS", 1)  # 1 مناسب للـ GPU الواحد
MAX_PENDING = _env_int("MAX_PENDING", 20)
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 25)  # حد الديسكورد للملفات العادية
MAX_DOWNLOAD_MB = _env_int("MAX_DOWNLOAD_MB", 500)  # حد الملف اللي بيتحمّل من لينك


class WhitenBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.queue = JobQueue(
            self._process_job,
            workers=QUEUE_WORKERS,
            max_pending=MAX_PENDING,
            listener=self._on_job_update,
        )

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession()
        self.queue.start()
        if CLEAN_ENGINE == "builtin":
            # نحمّل موديل الكشف من أول تشغيل بدل ما أول فصل يستنى
            asyncio.create_task(self._warm_model())
        if GUILD_ID:
            guild = discord.Object(id=GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def _warm_model(self) -> None:
        try:
            await ensure_model(MODEL_DIR)
        except Exception:
            log.exception("Could not download the text detector model")

    async def close(self) -> None:
        await self.queue.stop()
        if self.http_session is not None:
            await self.http_session.close()
        await super().close()

    async def on_ready(self) -> None:
        log.info("Logged in as %s (engine: %s)", self.user, CLEAN_ENGINE)

    # ---------- الشغل الفعلي ----------
    async def _process_job(self, job: Job) -> str:
        assert self.http_session is not None
        started = time.monotonic()
        if CLEAN_ENGINE == "remote":
            if not WORKER_URL:
                raise RuntimeError("CLEAN_ENGINE=remote but WORKER_URL is not set")
            cleaned_zip = await clean_chapter_remote(
                job.payload,
                profile=job.profile,
                worker_url=WORKER_URL,
                token=WORKER_TOKEN or "",
            )
        elif CLEAN_ENGINE == "cli":
            cleaned_zip = await clean_chapter_cli(
                job.payload,
                profile=job.profile,
                repo_dir=TRANSLATOR_REPO,
                python_exe=TRANSLATOR_PYTHON,
                work_root=WORK_ROOT,
            )
        else:
            cleaned_zip = await clean_chapter_local(
                job.payload,
                profile=job.profile,
                model_dir=MODEL_DIR,
                concurrency=PAGE_CONCURRENCY,
            )
        whiten_time = time.monotonic() - started
        log.info("Job #%s whitened in %.1fs (%s)", job.id, whiten_time, job.profile)
        return await upload_to_gofile(
            self.http_session,
            cleaned_zip,
            f"{job.name}-clean.zip",
            token=GOFILE_TOKEN,
        )

    async def _on_job_update(self, job: Job) -> None:
        if job.status not in (DONE, FAILED, CANCELLED):
            return
        channel = self.get_channel(job.channel_id) or await self.fetch_channel(job.channel_id)
        mention = f"<@{job.owner_id}>"
        if job.status == DONE:
            text = (
                f"{mention} ✅ Chapter **{job.name}** whitened (job #{job.id}, {job.profile}) "
                f"in {job.duration:.1f}s\n🔗 {job.result_url}"
            )
        elif job.status == FAILED:
            text = f"{mention} ❌ Job #{job.id} (**{job.name}**) failed: {job.error}"
        else:
            text = f"{mention} ⏹️ Job #{job.id} (**{job.name}**) was cancelled"
        await channel.send(text)  # type: ignore[union-attr]

    # ---------- الأوامر ----------
    def is_admin(self, member: discord.abc.User) -> bool:
        roles = getattr(member, "roles", [])
        return ADMIN_ROLE_ID != 0 and any(role.id == ADMIN_ROLE_ID for role in roles)


bot = WhitenBot()


def _channel_allowed(channel_id: int) -> bool:
    return not ALLOWED_CHANNEL_IDS or channel_id in ALLOWED_CHANNEL_IDS


@bot.tree.command(name="clean", description="Send a chapter zip (file or link) and get a gofile link with the whitened pages")
@app_commands.describe(
    chapter="Zip file with the chapter's images",
    link="Or a link: Google Drive, gofile, or a direct zip URL",
    profile="Quality: perfect or fast",
)
@app_commands.choices(profile=[
    app_commands.Choice(name="Perfect (best quality)", value="perfect"),
    app_commands.Choice(name="Fast (quick editing)", value="fast"),
])
async def clean_command(
    interaction: discord.Interaction,
    chapter: Optional[discord.Attachment] = None,
    link: Optional[str] = None,
    profile: Optional[app_commands.Choice[str]] = None,
) -> None:
    if not _channel_allowed(interaction.channel_id):
        await interaction.response.send_message("This command is not allowed in this channel.", ephemeral=True)
        return
    if (chapter is None) == (link is None):
        await interaction.response.send_message("Send either a **file** or a **link**, not both.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)
    chosen = profile.value if profile else PROFILE_CHOICES[0]

    try:
        if chapter is not None:
            if not chapter.filename.lower().endswith(".zip"):
                raise SourceError("The file must be a **.zip**")
            if chapter.size > MAX_UPLOAD_MB * 1024 * 1024:
                raise SourceError(f"The file is larger than {MAX_UPLOAD_MB} MB")
            filename, payload = chapter.filename, await chapter.read()
        else:
            filename, payload = await download_zip(
                bot.http_session,
                link,
                max_bytes=MAX_DOWNLOAD_MB * 1024 * 1024,
                gofile_token=GOFILE_TOKEN,
            )
    except (SourceError, aiohttp.ClientError) as exc:
        await interaction.followup.send(f"❌ {exc}", ephemeral=True)
        return

    name = os.path.splitext(filename)[0] or "chapter"
    try:
        job = await bot.queue.submit(
            owner_id=interaction.user.id,
            owner_name=str(interaction.user),
            channel_id=interaction.channel_id,
            name=name,
            profile=chosen,
            payload=payload,
        )
    except QueueFullError as exc:
        await interaction.followup.send(f"❌ {exc}", ephemeral=True)
        return

    position = await bot.queue.position(job)
    ahead = f"{position - 1} chapter(s) ahead of you" if position and position > 1 else "it's your turn now"
    await interaction.followup.send(
        f"📥 Added to the queue — job **#{job.id}** — {ahead} — quality: `{chosen}`"
    )


@bot.tree.command(name="queue", description="Show chapters being whitened and waiting")
async def queue_command(interaction: discord.Interaction) -> None:
    running, waiting = bot.queue.snapshot()
    lines = []
    for job in running:
        lines.append(f"⚙️ #{job.id} **{job.name}** — running ({job.profile}) — by {job.owner_name}")
    for index, job in enumerate(waiting, start=1):
        lines.append(f"⏳ {index}. #{job.id} **{job.name}** ({job.profile}) — by {job.owner_name}")
    text = "\n".join(lines) if lines else "Queue is empty ✨"
    await interaction.response.send_message(text)


@bot.tree.command(name="cancel", description="Cancel a chapter that is still waiting")
@app_commands.describe(job_id="Job ID (from /queue)")
async def cancel_command(interaction: discord.Interaction, job_id: int) -> None:
    try:
        job = await bot.queue.cancel(job_id, interaction.user.id, bot.is_admin(interaction.user))
    except (KeyError, PermissionError, ValueError) as exc:
        await interaction.response.send_message(f"❌ {exc}", ephemeral=True)
        return
    await interaction.response.send_message(f"⏹️ Cancelled job #{job.id}")


def main() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit("Set DISCORD_TOKEN in the .env file first")
    bot.run(DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
