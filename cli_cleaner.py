"""تبييض الفصل بالـ CLI بتاع manga-translator-ui (وضع inpaint-only) على فولدر الفصل كله في تشغيل واحد."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

from profiles import build_config
from whitening import build_zip, extract_pages, output_names

log = logging.getLogger("whiten-bot.cli")


class CliCleanError(RuntimeError):
    pass


async def clean_chapter_cli(
    zip_bytes: bytes,
    *,
    profile: str,
    repo_dir: str,
    python_exe: str,
    work_root: str,
    timeout_s: int = 3600,
) -> bytes:
    pages = extract_pages(zip_bytes)
    out_names = output_names([name for name, _ in pages])

    Path(work_root).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="chapter-", dir=work_root))
    try:
        in_dir = work / "in"
        out_dir = work / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # أسماء بسيطة بالترتيب عشان نرجع نربط كل ملف بصفحته (تجنب تكرار الأسماء)
        for index, (name, data) in enumerate(pages):
            suffix = Path(name).suffix.lower() or ".png"
            (in_dir / f"{index:04d}{suffix}").write_bytes(data)

        config = build_config(profile)
        config["cli"] = {"inpaint_only": True, "use_gpu": True}
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        cmd = [
            python_exe, "-m", "manga_translator", "local",
            "-i", str(in_dir), "-o", str(out_dir), "--config", str(config_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise CliCleanError("whitening timed out")

        if proc.returncode != 0:
            tail = output.decode("utf-8", "replace")[-500:]
            raise CliCleanError(f"CLI exited with code {proc.returncode}: {tail}")

        results: dict[int, Path] = {}
        for path in out_dir.rglob("*"):
            if path.is_file() and path.stem.isdigit():
                results[int(path.stem)] = path

        cleaned = []
        for index in range(len(pages)):
            source = results.get(index)
            if source is None:
                # الـ CLI بيتخطى الصفحات اللي مفيهاش نص مكتشف، فبنرجع الصفحة الأصلية زي ما هي
                log.warning("No CLI output for page %s; keeping the original", pages[index][0])
                original = next((in_dir.glob(f"{index:04d}.*")), None)
                if original is None:
                    raise CliCleanError(f"lost page {pages[index][0]}")
                cleaned.append((Path(out_names[index]).with_suffix(original.suffix.lower()).name, original.read_bytes()))
                continue
            out_name = Path(out_names[index]).with_suffix(source.suffix.lower())
            cleaned.append((out_name.name, source.read_bytes()))
        return build_zip(cleaned)
    finally:
        shutil.rmtree(work, ignore_errors=True)
