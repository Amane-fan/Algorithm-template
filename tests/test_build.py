from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from templatebook.cli import compile_pdf, publish_pdf, snapshot
from templatebook.export import build, render
from templatebook.project import BuildError, glob_match, load_project


FIXTURES = Path(__file__).parent / "fixtures"


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "templates").mkdir()
        self.config = self.root / "templates.toml"
        self.config.write_text('source = "templates"\noutput = "dist"\n', encoding="utf-8")

    def file(self, name, text):
        path = self.root / "templates" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path

    def configure(self, content):
        self.config.write_text('source = "templates"\noutput = "dist"\n' + content, encoding="utf-8")

    def test_exclude_globs_apply_to_both_outputs(self):
        self.file("Math/keep.cpp", "int KEEP = 1;\n")
        self.file("Math/skip.cpp", "int SECRET = 2;\n")
        self.file("Math/deep/draft.py", "SECRET = 3\n")
        self.file("Java/Main.java", "class SECRET {}\n")
        self.configure('exclude = ["Math/skip.cpp", "**/draft*", "Java/"]\n')
        project = load_project(self.config)
        md, typ = render(project)
        self.assertNotIn("SECRET", md)
        self.assertNotIn("skip.cpp", typ)
        self.assertNotIn("draft", typ)
        self.assertNotIn("Java", typ)
        self.assertEqual([e.key for e in project.modules], ["Math/keep.cpp"])
        self.assertTrue(glob_match("draft.cpp", "**/draft*"))
        self.assertFalse(glob_match("Math/deep/a.cpp", "Math/*.cpp"))

    def test_alias_order_nested_and_stable_auto_discovery(self):
        self.file("Math/a10.cpp", "10\n")
        self.file("Math/a2.cpp", "2\n")
        self.file("Math/Nested/a.cpp", "1\n")
        self.configure('[aliases]\n"Math" = "数学"\n"Math/a10.cpp" = "优先模块"\n[order]\n"Math" = ["a10.cpp"]\n')
        project = load_project(self.config)
        self.assertEqual(project.entries[0].title, "数学")
        self.assertEqual(project.entries[0].children[0].title, "优先模块")
        self.assertEqual(next(e for e in project.modules if e.key.endswith("Nested/a.cpp")).level, 3)
        self.file("Math/a3.cpp", "3\n")
        self.assertEqual([p.path.name for p in load_project(self.config).modules], ["a10.cpp", "a.cpp", "a2.cpp", "a3.cpp"])

    def test_codes_are_verbatim_and_build_is_deterministic(self):
        code = '// 中文\r\nstring s = "```";\r\n\tint x = 2;\r\n'
        path = self.file("Code/a.cpp", code)
        digest = hashlib.sha256(path.read_bytes()).digest()
        project = load_project(self.config)
        build(project)
        first = (project.output / "templates.md").read_bytes()
        build(project)
        self.assertEqual(first, (project.output / "templates.md").read_bytes())
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).digest())
        self.assertIn("````cpp", first.decode())

    def test_markdown_images_tables_math_and_reference_isolation(self):
        self.file("Notes/intro.md", '# 标题\n\n**文字** 和 $x^2$。\n\n![图][same]\n\n[same]: <images/a b.svg>\n\n| a | b |\n| --- | ---: |\n| 1 | 2 |\n\n$$\n\\sum_{i=1}^n i\n$$\n')
        project = load_project(self.config)
        md, typ = render(project)
        self.assertIn("#### 标题", md)
        self.assertIn("../templates/Notes/images/a%20b.svg", md)
        self.assertNotIn("[same]:", md)
        self.assertIn("| 1 | 2 |", md)
        self.assertIn("\\sum_{i=1}^n i", md)
        self.assertIn("cmarker.render", typ)
        self.assertIn("math: mitex", typ)

    def test_typst_includes_are_native_and_languages_correct(self):
        self.file("Mixed/native.typ", "$ sum_(i=1)^n i $\n")
        self.file("Mixed/main.java", "class Main {}\n")
        self.file("Mixed/solve.py", "print(1)\n")
        self.file("Mixed/run.sh", "echo ok\n")
        md, typ = render(load_project(self.config))
        for language in ("java", "python", "bash"):
            self.assertIn("```" + language, md)
            self.assertIn('lang: "' + language + '"', typ)
        self.assertIn('#include "../templates/Mixed/native.typ"', typ)
        self.assertIn("```typst", md)

    def test_bad_markdown_keeps_previous_outputs(self):
        path = self.file("Notes/a.md", "valid\n")
        project = load_project(self.config)
        build(project)
        original = (project.output / "templates.md").read_bytes()
        path.write_text("hello\n\n<div>unsupported</div>\n", encoding="utf-8")
        with self.assertRaisesRegex(BuildError, r"a.md:3"):
            build(load_project(self.config))
        self.assertEqual(original, (project.output / "templates.md").read_bytes())

    def test_invalid_config_encoding_and_missing_cover(self):
        self.file("Code/a.cpp", "int x;\n")
        for content in ('exclude = "wrong"', '[layout]\ncode_size = -1', '[layout]\naccent = "blue"', '[cover]\nimage = "missing.png"', '[order]\n"." = ["Code", "Code"]'):
            with self.subTest(content=content):
                self.configure(content)
                with self.assertRaises(BuildError):
                    load_project(self.config)
        self.configure("")
        (self.root / "templates/Code/a.cpp").write_bytes(b"\xff")
        with self.assertRaisesRegex(BuildError, "UTF-8"):
            load_project(self.config)

    def test_snapshot_detects_module_addition_and_removal(self):
        self.file("Code/a.cpp", "1\n")
        project = load_project(self.config)
        before = snapshot(project)
        new = self.file("Code/b.cpp", "2\n")
        self.assertNotEqual(before, snapshot(project))
        new.unlink()
        self.assertEqual(before, snapshot(project))

    def test_pdf_publish_preserves_previous_file_when_locked(self):
        source = self.root / "compiled.pdf"
        source.write_bytes(b"new pdf")
        destination = self.root / "templates.pdf"
        destination.write_bytes(b"previous pdf")
        with patch.object(Path, "replace", side_effect=PermissionError("file is locked")):
            with self.assertRaisesRegex(BuildError, "旧 PDF 已保留"):
                publish_pdf(source, destination)
        self.assertEqual(destination.read_bytes(), b"previous pdf")
        self.assertFalse(list(self.root.glob(".templatebook-publish-*")))

    @unittest.skipUnless(os.name == "nt", "Windows file ACL regression")
    def test_pdf_publish_inherits_output_directory_permissions(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            source = Path(directory) / "private.pdf"
            source.write_bytes(b"new pdf")
            destination = self.root / "templates.pdf"
            publish_pdf(source, destination)
        self.assertEqual(destination.read_bytes(), b"new pdf")
        # Check the protected-DACL bit through the Windows API, independent of
        # localized icacls output or the current account's administrator status.
        import ctypes
        from ctypes import wintypes

        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        advapi.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi.GetFileSecurityW.restype = wintypes.BOOL
        needed = wintypes.DWORD()
        advapi.GetFileSecurityW(str(destination), 4, None, 0, ctypes.byref(needed))
        descriptor = ctypes.create_string_buffer(needed.value)
        self.assertTrue(advapi.GetFileSecurityW(str(destination), 4, descriptor, needed, ctypes.byref(needed)))
        advapi.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD)]
        advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
        control, revision = wintypes.WORD(), wintypes.DWORD()
        self.assertTrue(advapi.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)))
        self.assertFalse(control.value & 0x1000, "Published PDF must not retain the private temporary directory's protected ACL")

    def test_watch_rebuilds_and_recovers_from_invalid_config(self):
        path = self.file("Code/a.py", "print('first')\n")
        command = [sys.executable, "-u", str(Path(__file__).resolve().parent.parent / "build.py"), "watch", "--no-pdf", "--config", str(self.config)]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.wait(timeout=5))
        self.addCleanup(process.terminate)
        output = self.root / "dist/templates.md"

        def wait_for(fragment):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if output.exists() and fragment in output.read_text(encoding="utf-8"):
                    return
                time.sleep(0.1)
            self.fail(f"watch did not produce {fragment!r}")

        wait_for("first")
        path.write_text("print('second')\n", encoding="utf-8")
        wait_for("second")
        self.configure("exclude = [")
        time.sleep(1.2)
        self.assertIsNone(process.poll())
        self.configure("")
        path.write_text("print('recovered')\n", encoding="utf-8")
        wait_for("recovered")

    @unittest.skipUnless(os.environ.get("TEMPLATEBOOK_TEST_PDF") == "1", "set TEMPLATEBOOK_TEST_PDF=1 for real Typst compilation")
    def test_compile_mixed_formats_and_image_cover(self):
        import shutil
        for path in (FIXTURES / "templates").rglob("*"):
            if path.is_file():
                target = self.root / "templates" / path.relative_to(FIXTURES / "templates")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        self.configure('[cover]\nimage = "templates/Mixed/assets/diagram.svg"\n[layout]\nfont = ["Noto Sans SC"]\ncode_font = ["DejaVu Sans Mono", "Noto Sans SC"]\n')
        project = load_project(self.config)
        build(project)
        compile_pdf(project)
        pdf = project.output / "templates.pdf"
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        self.assertGreater(pdf.stat().st_size, 10000)
        # Keep a review copy only when explicitly requested by the local verification run.
        if destination := os.environ.get("TEMPLATEBOOK_QA_PDF"):
            shutil.copyfile(pdf, destination)
        previous = pdf.read_bytes()
        self.file("Mixed/broken.typ", "#undefined_function()\n")
        project = load_project(self.config)
        build(project)
        with self.assertRaises(BuildError):
            compile_pdf(project)
        self.assertEqual(previous, pdf.read_bytes())


if __name__ == "__main__":
    unittest.main()
