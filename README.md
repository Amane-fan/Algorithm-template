# Amane の Templates

从 `templates/` 自动收集算法模板，生成同名的 `templates.md`、`templates.typ` 和 `templates.pdf`。

目录就是专题，文件就是模块。新增文件无需手工修改汇编文档；标题别名、排序、排除和封面图片统一在 [`templates.toml`](templates.toml) 中设置。

## 快速开始

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 环境与依赖。项目兼容 Python **3.11+**，`.python-version` 默认选择 **3.13**，uv 会在需要时下载对应的 Python。生成 Markdown / Typst 不需要安装 Typst；编译 PDF 需要 **Typst 0.15+** 或相应版本的 Tinymist。

```sh
uv sync --locked                 # 按 uv.lock 创建 .venv 并安装依赖
uv run python build.py           # 生成 templates.md 和 templates.typ
uv run python build.py pdf       # 重新生成并编译 templates.pdf
uv run python build.py watch     # 监听修改，自动重新生成 Markdown、Typst、PDF
```

无需手动激活虚拟环境；`uv run` 会使用并同步项目的 `.venv`。[uv 的项目运行机制](https://docs.astral.sh/uv/guides/projects/)会根据 `pyproject.toml` 和 `uv.lock` 保持依赖一致。

生成结果默认位于项目根目录。用 PDF 阅读器打开 `templates.pdf` 即可预览，支持文件自动重载的阅读器可与 `watch` 配合。也可以在 VS Code 中打开 `templates.typ`，使用 Tinymist 预览；配合 `uv run python build.py watch --no-pdf`，新增、删除模板和修改配置也会反映在预览中。

其他命令：

```sh
uv run python build.py list                    # 查看实际收录的模块、标题和顺序
uv run python build.py check                   # 校验配置、编码和 Markdown 转换，不写输出
uv run python build.py watch --no-pdf           # 只监听生成 Markdown / Typst
uv run python build.py pdf --config book.toml   # 使用另一份配置
uv run python build.py pdf --compiler /path/to/typst
```

配置中的路径都相对于配置文件。以上命令在项目目录运行；从其他目录运行可使用 `uv run --directory /path/to/project python build.py`。`check` 不检查 Typst 语法与最终布局，完整验证使用 `pdf`。

## 项目结构

```text
templates/
  DataStructure/
    SegmentTree.cpp
  Graph/
  Math/
  String/
  Java/
  Others/
templates.toml            # 内容与排版配置
pyproject.toml            # Python 项目信息与依赖
uv.lock                   # uv 生成的依赖锁文件，纳入版本管理
.python-version           # 默认 Python 版本
build.py                  # 命令行入口
templatebook/             # 扫描、转换与构建逻辑
styles/book.typ           # Typst 排版规则
styles/code.tmTheme       # 适合浅色纸面的语法高亮
tests/                    # 转换测试和混合格式验收样例
```

支持嵌套专题目录。隐藏文件、隐藏目录、`__pycache__` 和图片等资源不单独成为模块；空目录及排除后为空的目录不显示。模板必须是 UTF-8，可带 BOM。构建程序只读取模板，不改写源文件。

## 标题、排序与排除

默认使用目录名和去掉扩展名的文件名作为标题。别名不会改变文件名，配置中的键始终使用原始路径和正斜杠 `/`。

```toml
# 顶层配置，放在第一个 [表] 之前。
exclude = [
  "Others/test.cpp",       # 单个模块
  "Java/",                # 整个专题
  "**/draft*",            # 任意深度的草稿文件或目录
]

[aliases]
"DataStructure" = "数据结构"
"DataStructure/SegmentTree.cpp" = "线段树"

[order]
"." = ["DataStructure", "Graph", "Math", "String", "Java", "Others"]
"DataStructure" = ["DSU.cpp", "Fenwick.cpp", "SegmentTree.cpp"]
```

`exclude` 同时作用于 Markdown 和 Typst/PDF；默认空列表，即全部收录。`*` 和 `?` 只匹配当前路径段，`**` 可以匹配零层或多层目录。匹配区分大小写；引用图片、Typst 辅助文件仍可由其他模块使用，排除控制的是独立模块收录。

`order` 只需列出想放在前面的直接子项。剩余内容按名称自然排序（`a2` 在 `a10` 前），目录优先；新增文件自动进入输出。不存在的别名、排序条目和没有匹配的排除规则会输出提示。

## 内容格式

| 源文件 | Markdown 输出 | Typst / PDF 输出 |
| --- | --- | --- |
| `.cpp` / `.h` / `.hpp` / `.cc` / `.cxx` | `cpp` 代码块 | C++ 高亮代码 |
| `.java` | `java` 代码块 | Java 高亮代码 |
| `.py` | `python` 代码块 | Python 高亮代码 |
| `.md` / `.markdown` | 正文，提升标题层级并修正相对资源路径 | 渲染 Markdown |
| `.typ` | Typst 源码块 | 原生包含并渲染 |
| `.sh` / `.c` / `.txt` | 对应代码块 | 对应高亮或纯文本 |

其他 UTF-8 文本文件会作为普通代码模块收录并提示，可通过 `[languages]` 扩展，例如 `".rs" = "rust"`。二进制文件应放在资源目录并配置排除，或使用已识别的图片扩展名。

Markdown 支持标题、强调、删除线、列表、引用、链接、本地图片、代码块、表格，以及 `$...$` 和 `$$...$$` LaTeX 数学公式。推荐将行间公式的 `$$` 各放在独立一行。HTML 标签会报出源文件和行号；普通 HTML 注释不显示。复杂 LaTeX 宏受 MiTeX 支持范围限制，编译失败会保留详细诊断，不以公式源码冒充渲染结果。

图片路径相对于其 Markdown 源文件，含空格时使用 `![说明](<assets/my image.png>)`。Typst 无法直接加载网络图片，请使用本地资源。模块中的内层标题不进入书籍目录，目录保持“专题 / 模块”结构。

`.typ` 文件应是可嵌入的**内容片段**：可以使用公式、表格、绘图、`#let`、`#import`、相对图片或 `#include`；书籍整体的 `#set document`、`#set page` 和封面规则集中放在 `styles/book.typ`。片段内的标题自动下移层级，不重复编号。

## 封面与排版

```toml
[book]
title = "Amane の Templates"
subtitle = "Algorithms & Data Structures"
author = "Amane"

[cover]
image = "assets/cover.jpg"   # PNG / JPEG / SVG，留空为文字封面

[layout]
code_size = 8.0             # pt
code_leading = 2.2          # pt
keep_short_code_lines = 28 # 不超过此行数的代码块保持完整
body_size = 9.5             # pt
toc_columns = 2             # 1 或 2
section_new_page = false
accent = "315C61"
```

编辑已有表内的配置项，不要重复声明同名 TOML 表。封面图片位于标题下方，按固定区域居中裁切；未提供图片也能生成完整封面。

默认 A4 单栏正文、双栏目录、适合双面装订的内外边距、专题页眉和页码。封面不显示页码，目录使用罗马数字，正文从 1 重新编号。目录包含可点击的跳转和正文页码。长代码自动折行、跨页；不显示行号，短代码块尽量整块保留。

默认字体为 `Noto Sans SC`、`Microsoft YaHei` 和 `Maple Mono`。换电脑时请安装这些字体或修改 `layout.font` / `layout.code_font`；中文字体必须覆盖源文件使用的字符。Typst 自带的 `DejaVu Sans Mono` 可作为代码字体，但中文仍需额外字体。完整布局参数见 `templates.toml`，更细的设计调整在 `styles/book.typ`。

## 编译器、依赖与离线使用

Python 的 Markdown 转换依赖 Mistune，由 `pyproject.toml` 声明并通过 `uv.lock` 锁定。项目作为本地应用运行，不构建或安装自身的 Python 包。`.venv/` 已加入 Git 忽略规则，`pyproject.toml`、`uv.lock` 和 `.python-version` 应一起提交。

新增或删除 Python 依赖使用 `uv add 包名` / `uv remove 包名`，更新后同步提交 `pyproject.toml` 和 `uv.lock`。需要更新 Mistune 时显式执行 `uv add mistune==目标版本`，再运行测试。Typst 编译器和字体仍由系统提供，不属于 Python 依赖。

首次 `uv sync --locked` 需要获取 Python（若本机没有匹配版本）和依赖；环境准备好后可使用 `uv run --offline python build.py`。离线生成 PDF 时还需提前准备下面的 Typst 包缓存。

Typst 的 Markdown 渲染使用固定版本的 [cmarker 0.1.10](https://typst.app/universe/package/cmarker/)，数学公式使用 [MiTeX 0.2.7](https://typst.app/universe/package/mitex/)。只有收录 Markdown 模块时才加载这两个包。首次编译此类模块需要联网下载，随后使用 `.cache/typst/` 中的缓存；可通过 `TYPST_PACKAGE_CACHE_PATH` 指向已有缓存。纯代码和普通 Typst 模块不需要这两个包。

编译器搜索顺序：`--compiler` → `toolchain.compiler` → `TYPST_BIN` → PATH 中的 `typst` / `tinymist` → VS Code 的 Tinymist 扩展。也可在配置中固定路径：

```toml
[toolchain]
compiler = 'C:\path\to\typst.exe'
```

生成的 `templates.typ` 引用源文件和旁边自动生成的 `.templatebook-style.typ`、`.templatebook-code.tmTheme`。**转移 Typst 工程时应携带整个项目**；PDF 可以单独分发。`output = "dist"` 可将产物放入独立目录，但输出目录不能位于模板目录内。

PDF 先编译到临时目录，成功后通过输出目录中的暂存文件替换旧文件；最终文件继承输出目录权限，编译错误不会破坏上一次成功的 PDF。Windows 阅读器若锁定文件，关闭它后重试。

若旧版生成的 PDF 在替换时出现 `WinError 5`，先检查文件占用及只读属性。旧版临时目录权限也可能导致此问题，可以在项目目录执行 `icacls templates.pdf /reset`，使该文件恢复继承目录权限，再运行 `uv run python build.py pdf`。

## 验证

```sh
uv run --locked python -m unittest discover -s tests -v
```

测试覆盖统一排除、别名和排序、嵌套专题、原文保持、Markdown 表格/公式/图片、不同语言、非法配置、失败保留旧输出以及文件变化检测。

设置 `TEMPLATEBOOK_TEST_PDF=1` 后会额外编译混合格式测试文件，验证封面图片、原生 Typst 与 PDF 失败保护。例如 PowerShell：

```powershell
$env:TEMPLATEBOOK_TEST_PDF = '1'
$env:TYPST_PACKAGE_CACHE_PATH = "$PWD/.cache/typst"
uv run --locked python -m unittest discover -s tests -v
```
