"""Offline lifecycle and engine-profile regressions; no real Docker, SSH or GPU calls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def shell(self, text, **settings):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TF_', 'TENSORFOLD_', 'ENGINE', 'CONTEXT', 'KV_DTYPE', 'QUANT', 'MODEL_'))}
        env.update(settings)
        return subprocess.run(['bash', '-eu', '-c', text], cwd=ROOT, env=env,
                              capture_output=True, text=True)

    def test_profiles_keep_legacy_features_and_isolate_state(self):
        text = ('source scripts/config.sh; validate_engine; '
                'printf "%s\\n" "$CONTEXT" "$KV_DTYPE" "$CONTAINER_NAME" "$STATE_DIR" '
                '"$SERVED_NAME" "$KERNELS_PATH" "$(image_hash)"')
        old = self.shell(text, ENGINE='legacy')
        new = self.shell(text, ENGINE='native')
        self.assertEqual(old.returncode, 0, old.stderr)
        self.assertEqual(new.returncode, 0, new.stderr)
        a, b = old.stdout.splitlines(), new.stdout.splitlines()
        self.assertEqual(a[:2], ['1048576', 'fp8'])
        self.assertEqual(b[:2], ['262144', 'bf16'])
        self.assertNotEqual(a[2:4], b[2:4])
        self.assertEqual(a[4], 'nvidia/Qwen3.8-Flash-Next-NVFP4')
        self.assertEqual(a[4], b[4])
        self.assertTrue(a[5])
        self.assertEqual(b[5], '')
        self.assertNotEqual(a[6], b[6])
        result = self.shell('source scripts/config.sh; validate_engine', ENGINE='legacy', QUANT='int4ar')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_refuses_incompatible_settings_before_operations(self):
        settings = [dict(KV_DTYPE='fp8'), dict(CONTEXT='1048576'), dict(QUANT='int4ar'),
                    dict(TF_FLASHNEXT_KEEP_LATE='1'), dict(TF_FLASHNEXT_KV_PAGES='256'),
                    dict(TF_FLASHNEXT_DECODE_GRAPH='1'), dict(TENSORFOLD_IMAGE_TOKENS='16384'),
                    dict(MAX_VIDEOS='4'), dict(TF_REF='main')]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for name in ('docker', 'ssh'):
                p = tmp / name
                p.write_text('#!/bin/sh\necho forbidden >> "$CALL_LOG"\nexit 99\n')
                p.chmod(0o755)
            for setting in settings:
                for entry in ('start.sh', 'scripts/prepare.sh'):
                    with self.subTest(setting=setting, entry=entry):
                        result = self.shell(f'bash {entry}', ENGINE='native', WORKER='spark-worker',
                                            PATH=f'{td}:{os.environ["PATH"]}', CALL_LOG=str(tmp/'calls'), **setting)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('native', result.stderr)
                        self.assertFalse((tmp/'calls').exists())

    def test_extra_arguments_cannot_bypass_rank_or_context_checks(self):
        for args in ('--tp 1', '--rank=1', '--context 1048576', '--context=1048576',
                     '--kv-cache=fp8', '--vision-image-tokens 16384'):
            with self.subTest(args=args):
                result = self.shell(f'bash start.sh {args}', ENGINE='native', WORKER='spark-worker')
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Preflight', result.stdout)

    def test_both_rank_commands_match_release_parser_without_loading(self):
        binary = os.environ.get('TF_TEST_BINARY')
        if not binary:
            self.skipTest('set TF_TEST_BINARY to the checked native x86_64 1.2.6 release')
        # The real launcher builds this array; stop before networking/preparation.
        start = (ROOT/'start.sh').read_text()
        begin = start.index('SERVE_ARGS=(--context')
        end = start.index('USER_ARGS=("$@")', begin)
        fn_start = start.index('rank_argv() {')
        fn_end = start.index('\n}', fn_start) + 2
        for rank in (0, 1):
            script = ('source scripts/config.sh; validate_engine; ' + start[begin:end] + '\n'
                      'USER_ARGS=(); MODEL_ARG=/nonexistent/argument-check; MASTER_ADDR=192.0.2.1\n' +
                      start[fn_start:fn_end] + f'\nrank_argv {rank}\nprintf "%s\\0" "${{RANK_ARGV[@]}}"')
            result = self.shell(script, ENGINE='native')
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = result.stdout.rstrip('\0').split('\0')
            check = subprocess.run([binary, *argv[1:]], text=True, capture_output=True,
                                   env={**os.environ, 'HF_HUB_OFFLINE': '1'})
            self.assertEqual(check.returncode, 1, check.stderr)
            self.assertIn('neither a directory', check.stderr)
            self.assertNotIn('--kv-dtype', argv)
            self.assertEqual(argv[argv.index('--tp') + 1], '2')

    def test_stop_health_layouts_and_malformed_responses(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for name, body in {
                'docker': '#!/bin/sh\necho "$*" >> "$CALL_LOG"\ncase "$1" in\nps) echo ldsands-flashnext-dual-native;;\ninspect) echo true;;\nesac\n',
                'curl': '#!/bin/sh\nprintf "%s" "$FAKE_HEALTH"\n',
                'ssh': '#!/bin/sh\nexit 255\n',
            }.items():
                p = tmp/name
                p.write_text(body)
                p.chmod(0o755)
            cases = [({'live': {'connections': 3}}, 3), ({'running': 2}, 2),
                     ({'requests_running': 4}, 4), ({'running': True}, 0),
                     ({'live': {'connections': 'not-an-int'}}, 0), ([], 0)]
            for health, expected in cases:
                with self.subTest(health=health):
                    calls = tmp/'calls'
                    calls.write_text('')
                    result = self.shell('bash stop.sh', ENGINE='native', WORKER='', LOG_KEEP='0',
                                        PATH=f'{td}:{os.environ["PATH"]}', CALL_LOG=str(calls),
                                        FAKE_HEALTH=json.dumps(health))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('stop -t', calls.read_text())
                    self.assertIn('rm -f ldsands-flashnext-dual-native', calls.read_text())
                    if expected:
                        self.assertIn(f'{expected} request(s)', result.stderr)
                    else:
                        self.assertNotIn('request(s)', result.stderr)

    def test_start_memory_reads_memavailable_without_localized_free(self):
        lines = [s for s in (ROOT/'start.sh').read_text().splitlines()
                 if s.startswith(('here_gb=', 'there_gb='))]
        # Exercise the remote command's quoting too, locally with a fake worker.
        script = 'worker() { shift; bash -c "$*"; };\n' + '\n'.join(lines)
        script += '\n[[ "$here_gb" =~ ^[0-9]+$ && "$there_gb" =~ ^[0-9]+$ ]]'
        result = self.shell(script)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
