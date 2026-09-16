from __future__ import annotations

import fnmatch
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


class BuildError(Exception):
    pass


LANGUAGES = {
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".c": "c", ".java": "java", ".py": "python", ".sh": "bash",
    ".txt": "text", ".md": "markdown", ".markdown": "markdown", ".typ": "typst",
}
ASSETS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".pdf", ".ttf", ".otf"}


def natural_key(value: str):
    return [int(p) if p.isdigit() else p.casefold() for p in re.split(r"(\d+)", value)]


def glob_match(path: str, pattern: str) -> bool:
    """Path globs: * stays within a segment; ** matches zero or more segments."""
    parts, pats = path.split("/"), pattern.split("/")

    def match(i, j):
        if j == len(pats):
            return i == len(parts)
        if pats[j] == "**":
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], pats[j]) and match(i + 1, j + 1)

    return match(0, 0)


@dataclass
class Entry:
    path: Path
    key: str
    title: str
    level: int
    children: list[Entry] = field(default_factory=list)
    language: str | None = None
    text: str = ""

    @property
    def directory(self):
        return self.language is None

    @property
    def anchor(self):
        # Stable, unique even when aliases or filenames repeat.
        return "module-" + self.key.encode("utf-8").hex()

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Project:
    config_path: Path
    config: dict
    source: Path
    output: Path
    entries: list[Entry]
    warnings: list[str]

    @property
    def root(self):
        return self.config_path.parent

    def walk(self):
        for entry in self.entries:
            yield from entry.walk()

    @property
    def modules(self):
        return [entry for entry in self.walk() if not entry.directory]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise BuildError(f"{path}: 文件必须使用 UTF-8 编码") from exc


def inside(root: Path, value: str, label: str) -> Path:
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise BuildError(f"{label}: 路径必须位于项目目录中: {value}")
    return result


def load_project(config_path: Path) -> Project:
    config_path = config_path.resolve()
    try:
        config = tomllib.loads(read_text(config_path))
    except tomllib.TOMLDecodeError as exc:
        raise BuildError(f"{config_path}: {exc}") from exc
    allowed = {"source", "output", "exclude", "book", "cover", "layout", "aliases", "order", "languages", "toolchain"}
    if unknown := config.keys() - allowed:
        raise BuildError(f"未知配置项: {', '.join(sorted(unknown))}")
    for key in ("source", "output"):
        if not isinstance(config.get(key, ""), str):
            raise BuildError(f"{key} 必须是字符串")
    for key in ("book", "cover", "layout", "aliases", "order", "languages", "toolchain"):
        if not isinstance(config.setdefault(key, {}), dict):
            raise BuildError(f"{key} 必须是 TOML 表")
    defaults = {
        "book": {"title": "Templates", "subtitle": "Algorithms & Data Structures", "author": ""},
        "cover": {"image": ""},
        "layout": {
            "paper": "a4", "font": ["Noto Sans SC"],
            "code_font": ["DejaVu Sans Mono"], "body_size": 9.5, "code_size": 8.0,
            "code_leading": 2.2, "keep_short_code_lines": 28, "margin_top": 17, "margin_bottom": 17,
            "margin_inside": 18, "margin_outside": 14, "accent": "315C61",
            "toc_columns": 2, "section_new_page": False,
        },
        "toolchain": {"compiler": ""},
    }
    for section, values in defaults.items():
        if unknown := config[section].keys() - values.keys():
            raise BuildError(f"{section}: 未知配置项 {', '.join(sorted(unknown))}")
        for key, default in values.items():
            value = config[section].setdefault(key, default)
            if isinstance(default, bool):
                valid = type(value) is bool
            elif isinstance(default, (int, float)):
                valid = type(value) in (int, float) and 0 < value < 100
            else:
                valid = isinstance(value, type(default))
            if not valid:
                raise BuildError(f"{section}.{key}: 值的类型或范围不正确")
    layout = config["layout"]
    if layout["toc_columns"] not in (1, 2) or type(layout["toc_columns"]) is not int:
        raise BuildError("layout.toc_columns 必须是 1 或 2")
    if type(layout["keep_short_code_lines"]) is not int:
        raise BuildError("layout.keep_short_code_lines 必须是正整数")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", layout["accent"]):
        raise BuildError("layout.accent 必须是六位十六进制颜色")
    for key in ("font", "code_font"):
        if not layout[key] or not all(isinstance(f, str) and f for f in layout[key]):
            raise BuildError(f"layout.{key} 必须是非空字体名称数组")
    for key, value in config["aliases"].items():
        if not isinstance(value, str) or not value.strip() or "\n" in value:
            raise BuildError(f"aliases.{key}: 别名必须是非空单行字符串")
    for key, value in config["order"].items():
        if not isinstance(value, list) or not all(isinstance(n, str) and "/" not in n and "\\" not in n for n in value):
            raise BuildError(f"order.{key}: 必须是直接子项名称数组")
        if len(value) != len(set(value)):
            raise BuildError(f"order.{key}: 存在重复名称")
    for ext, lang in config["languages"].items():
        if not ext.startswith(".") or not isinstance(lang, str) or not re.fullmatch(r"[\w+-]+", lang):
            raise BuildError(f"languages.{ext}: 需要扩展名和高亮语言名称")
        if ext.lower() in (".md", ".markdown", ".typ"):
            raise BuildError(f"languages.{ext}: 文档类型不能覆盖为代码语言")
    excludes = config.setdefault("exclude", [])
    if not isinstance(excludes, list) or not all(isinstance(p, str) and p for p in excludes):
        raise BuildError("exclude 必须是非空路径模式的数组（可为空数组）")
    for pattern in excludes:
        if "\\" in pattern or pattern.startswith("/") or ".." in PurePosixPath(pattern).parts:
            raise BuildError(f"exclude: 使用相对于 source 的正斜杠路径: {pattern}")
    root = config_path.parent
    source = inside(root, config.get("source", "templates"), "source")
    output = inside(root, config.get("output", "."), "output")
    if not source.is_dir() or source == root:
        raise BuildError(f"source 必须是项目下存在的子目录: {source}")
    if output.is_relative_to(source):
        raise BuildError("output 不能位于 source 中")
    if config["cover"]["image"]:
        cover = inside(root, config["cover"]["image"], "cover.image")
        if not cover.is_file():
            raise BuildError(f"封面图片不存在: {cover}")
    warnings = []
    languages = LANGUAGES | {k.lower(): v for k, v in config["languages"].items()}
    matched = set()

    def excluded(key):
        result = False
        for pattern in excludes:
            p = pattern.rstrip("/")
            if glob_match(key, p) or glob_match(key, p + "/**"):
                matched.add(pattern)
                result = True
        return result

    def scan(directory, level):
        key = directory.relative_to(source).as_posix()
        order = config["order"].get(key, [])
        children = list(directory.iterdir())
        for name in order:
            if not any(p.name == name for p in children):
                warnings.append(f"order.{key}: 子项不存在: {name}")
        children.sort(key=lambda p: (0, order.index(p.name)) if p.name in order else (1, not p.is_dir(), natural_key(p.name)))
        result = []
        for path in children:
            relative = path.relative_to(source).as_posix()
            if path.name.startswith(".") or path.name == "__pycache__" or excluded(relative):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise BuildError(f"不支持模板符号链接: {relative}")
            if path.is_dir():
                sub = scan(path, level + 1)
                if sub:
                    result.append(Entry(path, relative, config["aliases"].get(relative, path.name), level, sub))
                continue
            if path.suffix.lower() in ASSETS:
                continue
            language = languages.get(path.suffix.lower(), "text")
            if path.suffix.lower() not in languages:
                warnings.append(f"{relative}: 未知文件类型，以普通代码块收录；可在 exclude / languages 配置")
            content = read_text(path)
            if "\x00" in content:
                raise BuildError(f"{relative}: 二进制文件不能作为模板，请配置 exclude")
            result.append(Entry(path, relative, config["aliases"].get(relative, path.stem), level, language=language, text=content))
        return result

    entries = scan(source, 1)
    for pattern in excludes:
        if pattern not in matched:
            warnings.append(f"exclude 未匹配任何文件或目录: {pattern}")
    for key in config["aliases"]:
        if not (source / key).exists():
            warnings.append(f"aliases 路径不存在: {key}")
    if not entries:
        raise BuildError("没有可收录的模板，请检查 source / exclude")
    return Project(config_path, config, source, output, entries, warnings)
