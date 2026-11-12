"""Release packaging and real grammar ABI checks, without models, GPUs or services."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_HASHES = {
    'aarch64': '250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b',
    'x86_64': '319f548ba3d8af501b30108fceccf6d12a375e2851999ae9f5cb88466406a694',
}


class NativePackagingTests(unittest.TestCase):
    def test_profile_and_docker_keep_verified_grammar_package(self):
        result = subprocess.run(
            ['bash', '-eu', '-c', 'source scripts/config.sh; validate_engine; '
             'printf "%s\\n" "$TF_REF" "$TF_RELEASE_VERSION" "$TF_RELEASE_SHA256"'],
            cwd=ROOT, env={**os.environ, 'ENGINE': 'native'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            '93a2e93ecebff8aa52b88e58c5dd222516e69c5f', '1.2.7', RELEASE_HASHES['aarch64']])
        dockerfile = (ROOT/'docker/native.Dockerfile').read_text()
        self.assertIn('sha256sum -c SHA256SUMS', dockerfile)
        self.assertIn('test -s lib/libtfgrammar.so', dockerfile)
        self.assertNotIn('rm -rf /opt/tensorfold/native/lib', dockerfile)
        prepare = (ROOT/'scripts/prepare.sh').read_text()
        self.assertIn('--build-arg TF_RELEASE_SHA256="$TF_RELEASE_SHA256"', prepare)

    def test_official_archives_and_internal_grammar_manifest(self):
        archives = {arch: os.environ.get('TF_TEST_ARCHIVE_' + arch.upper())
                    for arch in RELEASE_HASHES}
        if not any(archives.values()):
            self.skipTest('set TF_TEST_ARCHIVE_AARCH64 and/or TF_TEST_ARCHIVE_X86_64')
        for arch, path in archives.items():
            if not path:
                continue
            with self.subTest(arch=arch):
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), RELEASE_HASHES[arch])
                with tarfile.open(path) as archive:
                    files = {m.name.partition('/')[2]: m for m in archive.getmembers() if m.isfile()}
                    def read(name):
                        return archive.extractfile(files[name]).read()
                    self.assertEqual(read('VERSION').strip(), b'1.2.7')
                    manifest = {}
                    for line in read('SHA256SUMS').decode().splitlines():
                        digest, name = line.split(maxsplit=1)
                        manifest[name.removeprefix('./')] = digest
                    for name in ('bin/tensorfold-native', 'lib/libtfgrammar.so',
                                 'LICENSES/xgrammar-Apache-2.0.txt', 'LICENSES/xgrammar-NOTICE.txt'):
                        self.assertIn(name, manifest)
                    for name, digest in manifest.items():
                        self.assertEqual(hashlib.sha256(read(name)).hexdigest(), digest, name)
                    # Verify ELF machine values without executing the aarch64 artifacts.
                    for name in ('bin/tensorfold-native', 'lib/libtfgrammar.so'):
                        data = read(name)
                        self.assertEqual(data[:6], b'\x7fELF\x02\x01')
                        self.assertEqual(int.from_bytes(data[18:20], 'little'), 183 if arch == 'aarch64' else 62)


class NativeGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get('TF_TEST_BINARY')
        if not binary:
            raise unittest.SkipTest('set TF_TEST_BINARY to the verified x86_64 1.2.7 release')
        cls.binary = Path(binary).resolve()
        cls.lib = c.CDLL(str(cls.binary.parent.parent/'lib/libtfgrammar.so'))
        # ABI reviewed at the pinned release's zig/src/grammar/tf_grammar.cc.
        signatures = {
            'open': (c.c_void_p, [c.c_char_p, c.c_size_t, c.c_int, c.POINTER(c.c_int32), c.c_int, c.c_char_p, c.c_size_t]),
            'close': (None, [c.c_void_p]),
            'words': (c.c_int, [c.c_void_p]),
            'compile': (c.c_void_p, [c.c_void_p, c.c_int, c.c_char_p, c.c_size_t, c.c_char_p, c.c_size_t]),
            'free': (None, [c.c_void_p]),
            'matcher': (c.c_void_p, [c.c_void_p]),
            'matcher_free': (None, [c.c_void_p]),
            'accept': (c.c_int, [c.c_void_p, c.c_int32]),
            'fill': (c.c_int, [c.c_void_p, c.POINTER(c.c_int32), c.c_int]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(cls.lib, 'tfg_' + name)
            fn.restype, fn.argtypes = result, args
        cls.vocab = ['<eos>', *map(chr, range(32, 127))]

    def test_binary_discovers_adjacent_grammar_library(self):
        version = subprocess.run([self.binary, '--version'], capture_output=True, text=True, check=True)
        self.assertEqual(version.stdout.strip(), 'tensorfold-native 1.2.7')
        result = subprocess.run([self.binary, 'capabilities', '--json'], capture_output=True, text=True, check=True)
        grammar = next(x for x in json.loads(result.stdout)['optional_libraries'] if x['name'] == 'TensorFold grammar')
        self.assertTrue(grammar['found'])

    def test_strict_schema_accepts_typed_tool_arguments_and_rejects_invalid_output(self):
        tokenizer = json.dumps({'model': {'type': 'BPE', 'vocab': {v: i for i, v in enumerate(self.vocab)}},
                                'decoder': {'type': 'Fuse'}}).encode()
        error = c.create_string_buffer(1024)
        compiler = self.lib.tfg_open(tokenizer, len(tokenizer), len(self.vocab),
                                     (c.c_int32 * 1)(0), 1, error, len(error))
        self.assertTrue(compiler, error.value)
        try:
            schema = json.dumps({'type': 'object', 'properties': {
                'name': {'const': 'add_tags'}, 'arguments': {'type': 'object', 'properties': {
                    'doc_id': {'type': 'integer'}, 'tags': {'type': 'array', 'items': {'type': 'string'}}},
                    'required': ['doc_id', 'tags'], 'additionalProperties': False}},
                'required': ['name', 'arguments'], 'additionalProperties': False}).encode()
            grammar = self.lib.tfg_compile(compiler, 1, schema, len(schema), error, len(error))
            self.assertTrue(grammar, error.value)
            try:
                good = '{"name":"add_tags","arguments":{"doc_id":42,"tags":["urgent"]}}'
                for reply, expected in [(good, True), (good.replace('42', '"42"'), False),
                                        (good.replace('["urgent"]', '"urgent"'), False),
                                        (good.replace('add_tags', 'delete'), False),
                                        (good[:-1] + ',"extra":1}', False), (good[:-2], False)]:
                    with self.subTest(reply=reply):
                        matcher = self.lib.tfg_matcher(grammar)
                        self.assertTrue(matcher)
                        try:
                            accepted = all(self.lib.tfg_accept(matcher, self.vocab.index(char)) == 1 for char in reply)
                            words = self.lib.tfg_words(compiler)
                            mask = (c.c_int32 * words)()
                            self.assertEqual(self.lib.tfg_fill(matcher, mask, words), 0)
                            self.assertEqual(accepted and bool(mask[0] & 1), expected)
                        finally:
                            self.lib.tfg_matcher_free(matcher)
            finally:
                self.lib.tfg_free(grammar)
            bad = b'{"type":'
            self.assertFalse(self.lib.tfg_compile(compiler, 1, bad, len(bad), error, len(error)))
            self.assertTrue(error.value)
        finally:
            self.lib.tfg_close(compiler)


if __name__ == '__main__':
    unittest.main()
