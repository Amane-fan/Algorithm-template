from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .export import STYLE, build
from .project import BuildError, load_project, natural_key


def find_compiler(project, override=None):
    configured = override or project.config["toolchain"]["compiler"] or os.environ.get("TYPST_BIN")
    if configured:
        path = Path(configured).expanduser()
        local = project.root / path
        found = str(local.resolve()) if local.is_file() else shutil.which(configured)
        if not found:
            raise BuildError(f"找不到配置的编译器: {configured}")
        return found
    for name in ("typst", "tinymist"):
        if found := shutil.which(name):
            return found
    extensions = Path.home() / ".vscode" / "extensions"
    candidates = sorted(extensions.glob("myriad-dreamin.tinymist-*/out/tinymist*"), key=lambda p: natural_key(str(p)), reverse=True)
    for path in candidates:
        if path.name in ("tinymist", "tinymist.exe") and path.is_file():
            return str(path)
    raise BuildError("找不到 Typst 编译器。安装 Typst 0.15+，或设置 toolchain.compiler / TYPST_BIN / --compiler。")


def publish_pdf(source: Path, destination: Path):
    # TemporaryDirectory is private on Windows. Renaming its output directly
    # would retain that ACL, preventing other project users from updating it.
    # A newly created sibling inherits the output directory's normal permissions.
    staging = destination.with_name(f".templatebook-publish-{uuid.uuid4().hex}.pdf")
    try:
        with source.open("rb") as reader, staging.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        staging.replace(destination)
    except PermissionError as exc:
        raise BuildError(
            f"PDF 已编译，但无法替换 {destination}。请关闭占用它的阅读器，"
            "并检查文件是否只读、当前用户是否具有修改权限；旧 PDF 已保留。"
        ) from exc
    finally:
        staging.unlink(missing_ok=True)


def compile_pdf(project, override=None):
    compiler = find_compiler(project, override)
    cache = Path(os.environ.get("TYPST_PACKAGE_CACHE_PATH", project.root / ".cache" / "typst"))
    cache.mkdir(parents=True, exist_ok=True)
    # Keep the last successful PDF on errors; also avoids Tinymist's stem/directory collision.
    with tempfile.TemporaryDirectory(prefix=".templatebook-", dir=project.output) as directory:
        temporary = Path(directory) / "templates.pdf"
        result = subprocess.run([compiler, "compile", "--root", str(project.root), "--package-cache-path", str(cache.resolve()), str(project.output / "templates.typ"), str(temporary)], cwd=project.root)
        if result.returncode or not temporary.is_file():
            raise BuildError("PDF 编译失败。上方为 Typst 原始错误；templates.typ 的 source 注释可定位源模块。")
        publish_pdf(temporary, project.output / "templates.pdf")
    print(f"PDF: {project.output / 'templates.pdf'}")


def snapshot(project):
    paths = [project.config_path, STYLE, STYLE.parent / "code.tmTheme"]
    paths.extend(p for p in project.source.rglob("*") if p.is_file())
    # Local includes / images outside source also trigger preview rebuilds.
    paths.extend(p for p in project.root.rglob("*") if p.is_file() and
                 p.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".typ") and
                 not any(part.startswith(".") for part in p.relative_to(project.root).parts) and
                 p.name not in ("templates.typ",) and
                 not p.is_relative_to(project.root / "tmp"))
    return tuple(sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in set(paths) if p.exists()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="将目录中的算法模板构建为 templates.md / templates.typ / templates.pdf")
    parser.add_argument("command", nargs="?", default="build", choices=["build", "pdf", "watch", "check", "list"])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "templates.toml")
    parser.add_argument("--compiler", help="Typst / Tinymist 可执行文件路径")
    parser.add_argument("--no-pdf", action="store_true", help="watch 时只生成 Markdown / Typst")
    args = parser.parse_args(argv)
    try:
        project = load_project(args.config)
        for warning in project.warnings:
            print("提示: " + warning, file=sys.stderr)
        if args.command == "list":
            for entry in project.walk():
                print("  " * (entry.level - 1) + entry.title + f" [{entry.key}]")
            return 0
        if args.command == "check":
            from .export import render
            render(project)
            print(f"检查通过：{len(project.modules)} 个模块。语法和排版检查请运行 pdf。")
            return 0
        build(project)
        print(f"已生成 templates.md / templates.typ：{len(project.modules)} 个模块。")
        if args.command == "pdf" or (args.command == "watch" and not args.no_pdf):
            compile_pdf(project, args.compiler)
        if args.command != "watch":
            return 0
        print("正在监听模板、配置、样式和本地图片；Ctrl+C 退出。")
        previous = snapshot(project)
        while True:
            time.sleep(0.8)
            current = snapshot(project)
            if current == previous:
                continue
            previous = current
            try:
                project = load_project(args.config)
                build(project)
                if not args.no_pdf:
                    compile_pdf(project, args.compiler)
                previous = snapshot(project)
                print("已更新。", flush=True)
            except (BuildError, OSError) as exc:
                print(f"更新失败，修正后将重试: {exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\n已停止监听。")
        return 0
    except (BuildError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
