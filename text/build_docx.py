"""Сборка курсовой работы в формате DOCX."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

FILES = [
    "01_title.md",
    "02_intro.md",
    "03_chapter1.md",
    "04_chapter2.md",
    "05_conclusion.md",
    "06_references.md",
    "07_appendix.md",
]


def merge_markdown(text_dir: Path, output: Path) -> None:
    """Объединяет markdown-файлы в один."""
    with open(output, "w", encoding="utf-8") as out:
        for fname in FILES:
            fpath = text_dir / fname
            if not fpath.exists():
                logger.warning("Файл не найден: %s", fpath)
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                out.write(f.read())
                out.write("\n\n")
    logger.info("Объединённый Markdown сохранён: %s", output)


def build_docx(draft: Path, output: Path) -> None:
    """Конвертирует Markdown в DOCX через pandoc."""
    pandoc_path = r"C:\Windows\Temp\pandoc\pandoc-3.1.11\pandoc.exe"
    cmd = [
        pandoc_path,
        str(draft),
        "-o",
        str(output),
        "--from",
        "markdown",
        "--to",
        "docx",
        "--toc",
        "--toc-depth=2",
        "--highlight-style=tango",
        "-V",
        "lang=ru",
    ]
    logger.info("Запуск: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Pandoc ошибка: %s", result.stderr)
        raise RuntimeError(result.stderr)
    logger.info("DOCX собран: %s", output)


def main():
    logging.basicConfig(level=logging.INFO)
    text_dir = Path(__file__).parent
    draft = text_dir / "draft.md"
    output = text_dir.parent / "coursework.docx"
    merge_markdown(text_dir, draft)
    build_docx(draft, output)


if __name__ == "__main__":
    main()
