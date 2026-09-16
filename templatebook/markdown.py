"""Normalize Markdown for a combined document; parsing is delegated to Mistune."""
from __future__ import annotations

import os
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .project import BuildError


def fenced(text: str, language: str) -> str:
    marker = "`" * max(3, 1 + max((len(m.group()) for m in re.finditer(r"`+", text)), default=0))
    return f"{marker}{language}\n{text}" + ("" if text.endswith("\n") else "\n") + marker + "\n"


def escape_title(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", text)


def render_module(entry, project) -> str:
    try:
        import mistune
        from mistune.renderers.markdown import MarkdownRenderer
    except ImportError as exc:
        raise BuildError("Markdown 模块需要 mistune：请运行 uv sync，并使用 uv run python build.py 构建") from exc

    class Renderer(MarkdownRenderer):
        def heading(self, token, state):
            token = dict(token, attrs=dict(token["attrs"], level=min(6, token["attrs"]["level"] + entry.level + 1)))
            return super().heading(token, state)

        def link(self, token, state):
            # Resolve references independently per file, preventing duplicate [ref] definitions.
            token = dict(token, label=None, attrs=dict(token["attrs"]))
            url = urlsplit(token["attrs"]["url"])
            if url.path and not url.scheme and not url.netloc:
                path = (project.root / unquote(url.path).lstrip("/")) if url.path.startswith("/") else (entry.path.parent / unquote(url.path))
                relative = os.path.relpath(path, project.output).replace(os.sep, "/")
                token["attrs"]["url"] = urlunsplit(("", "", quote(relative, safe="/"), url.query, url.fragment))
            return super().link(token, state)

        def render_referrences(self, state):
            return iter(())

        def codespan(self, token, state):
            text = token["raw"]
            delimiter = "`" * (1 + max((len(m.group()) for m in re.finditer(r"`+", text)), default=0))
            pad = " " if text.startswith("`") or text.endswith("`") else ""
            return delimiter + pad + text + pad + delimiter

        def strikethrough(self, token, state):
            return "~~" + self.render_children(token, state) + "~~"

        def inline_math(self, token, state):
            return "$" + token["raw"] + "$"

        def block_math(self, token, state):
            return "$$\n" + token["raw"] + "\n$$\n\n"

        def table(self, token, state):
            return self.render_children(token, state) + "\n"

        def table_head(self, token, state):
            cells = token["children"]
            row = self.render_children(token, state)
            aligns = {None: "---", "left": ":---", "right": "---:", "center": ":---:"}
            return "| " + row + "\n| " + " | ".join(aligns[c["attrs"].get("align")] for c in cells) + " |\n"

        def table_body(self, token, state):
            return self.render_children(token, state)

        def table_row(self, token, state):
            return "| " + self.render_children(token, state) + "\n"

        def table_cell(self, token, state):
            return self.render_children(token, state).replace("|", "\\|") + " | "

        def inline_html(self, token, state):
            return self.check_html(token)

        def block_html(self, token, state):
            return self.check_html(token) + "\n\n"

        def check_html(self, token):
            raw = token["raw"]
            if raw.strip().startswith("<!--"):
                return ""
            line = entry.text[:entry.text.find(raw)].count("\n") + 1
            raise BuildError(f"{entry.path}:{line}: 不支持 HTML，请使用 Markdown 或独立 .typ 模块")

    renderer = Renderer()
    parser = mistune.create_markdown(renderer=renderer, plugins=["table", "math", "strikethrough"])
    try:
        return parser(entry.text)
    except (ValueError, KeyError) as exc:
        raise BuildError(f"{entry.path}: Markdown 转换失败: {exc}") from exc
