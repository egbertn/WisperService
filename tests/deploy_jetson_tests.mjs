import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const deploy = fileURLToPath(new URL('../scripts/deploy-jetson.sh', import.meta.url));

function runDeployment(failure = '') {
    const root = mkdtempSync(join(tmpdir(), 'whisper-deploy-test-'));
    try {
        const repository = join(root, 'repository with spaces');
        const prefix = join(root, 'private prefix');
        const bin = join(root, 'mock-bin');
        const log = join(root, 'commands.log');
        for (const directory of [
            join(repository, '.git'), join(prefix, 'bin'),
            join(prefix, 'deps/whisper'), join(prefix, 'deps/ffmpeg/lib/pkgconfig'), bin,
        ]) {
            mkdirSync(directory, { recursive: true });
        }
        const executable = join(prefix, 'bin/whisper_transcribe');
        writeFileSync(executable, 'old executable', { mode: 0o755 });
        writeFileSync(log, '');
        writeFileSync(join(bin, 'ssh'), '#!/usr/bin/env bash\nexec bash -c "$4"\n', { mode: 0o755 });
        const mock = join(bin, 'mock');
        writeFileSync(mock, `#!/usr/bin/env bash
set -euo pipefail
tool=\${0##*/}
printf '%s %s\\n' "$tool" "$*" >> "$TEST_LOG"
case "$tool" in
    git)
        case "$1" in
            rev-parse)
                case "$2" in
                    --show-toplevel) printf '%s\\n' "$TEST_REPOSITORY" ;;
                    --git-path) printf '%s/.git/deploy-jetson.lock\\n' "$TEST_REPOSITORY" ;;
                    --verify) echo 0123456789abcdef ;;
                esac ;;
            status) if [[ $TEST_FAILURE == dirty ]]; then echo ' M changed.cpp'; fi ;;
        esac ;;
    cmake)
        case "$1" in
            -S) [[ $TEST_FAILURE != configure ]] ;;
            --build) [[ $TEST_FAILURE != build ]] ;;
            --install)
                [[ $TEST_FAILURE != install ]] || exit 1
                mkdir -p "$DESTDIR$JETSON_PREFIX/bin"
                printf 'new executable' > "$DESTDIR$JETSON_PREFIX/bin/whisper_transcribe" ;;
        esac ;;
    ctest)
        [[ " $* " == *" --no-tests=error "* ]]
        [[ $TEST_FAILURE != tests && $TEST_FAILURE != no-tests ]] ;;
    sudo) shift; exec "$@" ;;
    systemctl)
        if [[ $1 == restart && $TEST_FAILURE == restart ]]; then exit 1; fi ;;
esac
`, { mode: 0o755 });
        for (const tool of ['git', 'cmake', 'ctest', 'sudo', 'systemctl', 'ninja', 'pkg-config']) {
            symlinkSync(mock, join(bin, tool));
        }
        const result = spawnSync('bash', [deploy, 'test@jetson', repository, 'main'], {
            encoding: 'utf8',
            timeout: 10000,
            env: {
                ...process.env,
                PATH: `${bin}:${process.env.PATH}`,
                JETSON_PREFIX: prefix,
                TEST_REPOSITORY: repository,
                TEST_LOG: log,
                TEST_FAILURE: failure,
                TMPDIR: root,
            },
        });
        assert.ifError(result.error);
        return {
            ...result,
            log: readFileSync(log, 'utf8'),
            executable: readFileSync(executable, 'utf8'),
        };
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

test('successful tests precede installation and service restart', () => {
    const result = runDeployment();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.executable, 'new executable');
    const tests = result.log.indexOf('ctest ');
    const install = result.log.indexOf('cmake --install ');
    const restart = result.log.indexOf('systemctl restart ');
    assert.ok(tests >= 0 && tests < install && install < restart, result.log);
    assert.match(result.log, /systemctl is-active --quiet whisper_transcribe\.service/);
    assert.match(result.stdout, /is active/);
});

for (const failure of ['dirty', 'configure', 'build', 'tests', 'no-tests', 'install']) {
    test(`${failure} failure preserves the installed executable and does not restart`, () => {
        const result = runDeployment(failure);
        assert.notEqual(result.status, 0);
        assert.equal(result.executable, 'old executable');
        assert.doesNotMatch(result.log, /systemctl restart /);
        const reached = {
            dirty: /git status /,
            configure: /cmake -S /,
            build: /cmake --build /,
            tests: /ctest /,
            'no-tests': /ctest /,
            install: /cmake --install /,
        };
        assert.match(result.log, reached[failure]);
        if (failure !== 'install') {
            assert.doesNotMatch(result.log, /cmake --install /);
        }
    });
}

test('restart failure is reported, not presented as success', () => {
    const result = runDeployment('restart');
    assert.notEqual(result.status, 0);
    assert.match(result.log, /systemctl restart /);
    assert.doesNotMatch(result.stdout, /is active/);
});
