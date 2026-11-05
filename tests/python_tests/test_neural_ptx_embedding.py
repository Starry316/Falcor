"""Exercise the actual PTX embedder and generated entry selector (no GPU needed).

Run in an MSVC developer shell:
python tests/python_tests/test_neural_ptx_embedding.py --embed-exe PATH_TO_neural_embed_ptx.exe
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

EMBED = None


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EMBED is None:
            raise unittest.SkipTest('Run explicitly with --embed-exe in an MSVC developer shell')

    def test_select_preserves_shared_code_and_handles_crlf_comments_and_strings(self):
        shared = '.version 8.0\n.target sm_75\n.address_size 64\n.visible .global .u32 shared;\n'
        first = '.visible /* between directives */ .entry first() {\n// } .entry fake() {}\n{ .reg .u32 r; }\nret;\n}'
        middle = '\n.func helper() { ret; }\n.file 1 "a{b}\\\"c"\n'
        second = '.visible .entry second() { /* } { */ ret; }'
        trailing = '\n// footer\n'
        text = shared + first + middle + second + trailing
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.ptx').write_bytes(text.replace('\n', '\r\n').encode())
            subprocess.run([str(EMBED), str(root / 'input.ptx'), str(root / 'embedded.h')], check=True)
            # Compile the generated header, rather than emulating its C++ logic.
            (root / 'check.cpp').write_text('''#include "embedded.h"
#include <fstream>
int main() {
    for (const auto& entry : mpmat_coop_entries)
        std::ofstream(std::string(entry.name) + ".ptx", std::ios::binary) << mpmat_coop_ptx_for_entry(entry.name);
    try { mpmat_coop_ptx_for_entry("missing"); return 1; }
    catch (const std::runtime_error&) { return 0; }
}
''')
            subprocess.run(['cl', '/nologo', '/std:c++17', '/EHsc', 'check.cpp', '/Fe:check.exe'], cwd=root, check=True)
            subprocess.run([str(root / 'check.exe')], cwd=root, check=True)
            self.assertEqual((root / 'first.ptx').read_bytes(), (shared + first + middle + trailing).encode())
            self.assertEqual((root / 'second.ptx').read_bytes(), (shared + middle + second + trailing).encode())

    def test_rejects_bad_entries_without_replacing_existing_header(self):
        for text in ('.version 8.0', '.entry x() {', '.entry x();',
                     '.entry x() {} .entry x() {}', '.entry x() { /* unfinished',
                     '.entry x() { "unfinished'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'input.ptx').write_text(text)
                output = root / 'output.h'
                output.write_text('previous good output')
                result = subprocess.run([str(EMBED), str(root / 'input.ptx'), str(output)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'previous good output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed-exe', required=True, type=Path)
    args, rest = parser.parse_known_args()
    EMBED = args.embed_exe.resolve()
    unittest.main(argv=[__file__, *rest])
