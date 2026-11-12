"""Offline FP8 wiring and reproducible source checks. No GPU or service operations."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = json.loads((ROOT/'native/inputs.json').read_text())
PATCH = ROOT/'native-patches/0001-flashnext-scaled-e4m3-cache.patch'


def shell(command, **settings):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('TF_', 'TENSORFOLD_', 'NATIVE_', 'ENGINE', 'KV_DTYPE', 'CONTEXT'))}
    return subprocess.run(['bash', '-eu', '-c', command], cwd=ROOT,
                          env={**env, **settings}, text=True, capture_output=True)


class Fp8WiringTests(unittest.TestCase):
    def test_precisions_select_different_executables_and_arguments(self):
        for dtype, binary, flags in [('bf16', 'tensorfold-native', []),
                                     ('fp8', 'tensorfold-native-fp8', ['--kv-cache', 'fp8'])]:
            with self.subTest(dtype=dtype):
                result = shell('source scripts/config.sh; validate_engine; SERVE_ARGS=(); '
                               'kv_arguments; engine_binary; printf "%s\\n" "${SERVE_ARGS[@]}"',
                               ENGINE='native', KV_DTYPE=dtype, NATIVE_FP8_EXPERIMENTAL='1')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [binary, *(flags or [''])])
        legacy = shell('source scripts/config.sh; SERVE_ARGS=(); kv_arguments; '
                       'printf "%s\\n" "${SERVE_ARGS[@]}"', ENGINE='legacy', KV_DTYPE='fp8')
        self.assertEqual(legacy.stdout.splitlines(), ['--kv-dtype', 'fp8'])

    def test_unqualified_fp8_requires_explicit_experimental_selection(self):
        result = shell('source scripts/config.sh; validate_engine', ENGINE='native', KV_DTYPE='fp8')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no GPU qualification', result.stderr)
        self.assertIn('NATIVE_FP8_EXPERIMENTAL=1', result.stderr)

    def test_fp8_preserves_other_defaults_and_refuses_invalid_profiles(self):
        result = shell('source scripts/config.sh; validate_engine; '
                       'printf "%s\\n" "$CONTEXT" "$PARALLEL"',
                       ENGINE='native', KV_DTYPE='fp8', NATIVE_FP8_EXPERIMENTAL='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['262144', '16'])
        for extra in [dict(TF_FLASHNEXT_KV_PAGES='64'), dict(TF_FLASHNEXT_DECODE_GRAPH='1'),
                      dict(TF_FLASHNEXT_WINDOW_GRAPH='1'), dict(TF_FLASHNEXT_KV_PAGE_TEST='64'),
                      dict(CONTEXT='262145'), dict(QUANT='int4ar')]:
            result = shell('source scripts/config.sh; validate_engine', ENGINE='native',
                           KV_DTYPE='fp8', NATIVE_FP8_EXPERIMENTAL='1', **extra)
            self.assertNotEqual(result.returncode, 0, extra)

    def test_rank_commands_and_parser_check_use_selected_executable(self):
        start = (ROOT/'start.sh').read_text()
        begin = start.index('rank_argv() {')
        end = start.index('\n}', begin) + 2
        for rank in (0, 1):
            result = shell('source scripts/config.sh; validate_engine; SERVE_ARGS=(); '
                           'kv_arguments; USER_ARGS=(); MODEL_ARG=/missing; MASTER_ADDR=192.0.2.1; '
                           + start[begin:end] + f'\nrank_argv {rank}; printf "%s\\n" "${{RANK_ARGV[@]}}"',
                           ENGINE='native', KV_DTYPE='fp8', NATIVE_FP8_EXPERIMENTAL='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.splitlines()
            self.assertEqual(args[0], 'tensorfold-native-fp8')
            self.assertEqual(args[args.index('--kv-cache') + 1], 'fp8')
            self.assertEqual(args[args.index('--rank') + 1], str(rank))
        self.assertIn('--entrypoint "/opt/tensorfold/native/bin/${RANK_ARGV[0]}"', start)

    def test_fp8_binary_parses_managed_flags_without_loading(self):
        binary = os.environ.get('TF_TEST_FP8_BINARY')
        if not binary:
            self.skipTest('set TF_TEST_FP8_BINARY to the source-built host verification executable')
        for rank in (0, 1):
            result = subprocess.run([binary, 'serve', '/nonexistent/argument-check', '--kv-cache', 'fp8',
                                     '--context', '262144', '--parallel', '16', '--tp', '2',
                                     '--rank', str(rank), '--master', '192.0.2.1', '--master-port', '29551'],
                                    text=True, capture_output=True,
                                    env={**os.environ, 'HF_HUB_OFFLINE': '1'})
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('neither a directory', result.stderr)

    def test_source_build_is_pinned_and_cannot_use_unpatched_tree(self):
        docker = (ROOT/'docker/native.Dockerfile').read_text()
        for value in [INPUTS['tensorfold_ref'], INPUTS['tensorfold_source_sha256'],
                      INPUTS['zig_version'], INPUTS['zig_aarch64_sha256']]:
            self.assertIn(value, docker)
        self.assertIn('COPY --from=fp8-build', docker)
        self.assertIn('bash /opt/build-native-fp8.sh', docker)
        self.assertIn('sha256sum -c SHA256SUMS', docker)
        builder = (ROOT/'scripts/build-native-fp8.sh').read_text()
        self.assertIn('-Dnvcc="$nvcc_bin" -Dsm=121', builder)
        self.assertNotIn('-Dfatbins=', builder)
        self.assertIn('native-patches/*.patch', (ROOT/'scripts/config.sh').read_text())
        self.assertIn('scripts/build-native-fp8.sh', (ROOT/'scripts/config.sh').read_text())
        self.assertIn('fn_kv8_abi_e4m3row1', PATCH.read_text())
        self.assertIn('kv_storage_abi', PATCH.read_text())
        self.assertEqual(INPUTS['head_bytes'], 256 + 4)


class PinnedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = os.environ.get('TF_TEST_SOURCE_ARCHIVE')
        if not archive:
            raise unittest.SkipTest('set TF_TEST_SOURCE_ARCHIVE to the pinned source archive')
        data = Path(archive).read_bytes()
        if hashlib.sha256(data).hexdigest() != INPUTS['tensorfold_source_sha256']:
            raise AssertionError('source archive checksum differs from the build pin')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        with tarfile.open(archive) as tf:
            tf.extractall(cls.root, filter='data')
        cls.source = cls.root / ('TensorFold-' + INPUTS['tensorfold_ref'])
        for args in [ ['--check'], [] ]:
            subprocess.run(['git', 'apply', *args, str(PATCH)], cwd=cls.source, check=True,
                           text=True, capture_output=True)

    def test_checked_patch_reproduces_the_reviewed_source(self):
        reviewed = os.environ.get('TF_TEST_PATCHED_SOURCE_DIR')
        if not reviewed:
            self.skipTest('set TF_TEST_PATCHED_SOURCE_DIR for byte-for-byte build source comparison')
        paths = [line.removeprefix('+++ b/') for line in PATCH.read_text().splitlines()
                 if line.startswith('+++ b/')]
        for path in paths:
            self.assertEqual((self.source/path).read_bytes(), (Path(reviewed)/path).read_bytes(), path)

    def test_actual_cuda_codec_on_cpu(self):
        include = os.environ.get('TF_TEST_CUDA_INCLUDE')
        if not include or not shutil.which('c++'):
            self.skipTest('set TF_TEST_CUDA_INCLUDE to CUDA headers and install a C++ compiler')
        out = self.root/'codec'
        subprocess.run(['c++', '-std=c++17', '-O2', '-I', include, '-I',
                        str(self.source/'zig/kernels/cuda'), str(ROOT/'native/kv8_contract.cpp'),
                        '-o', str(out)], check=True, capture_output=True, text=True)
        result = subprocess.run([out], check=True, text=True, capture_output=True)
        self.assertIn('E4M3 CPU contracts passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
