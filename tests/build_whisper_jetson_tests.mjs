import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('../scripts/build-whisper-jetson.sh', import.meta.url));

function runBuild(failure = '', overrides = {}, args = ['test@jetson'], setup = {}) {
    const root = mkdtempSync(join(tmpdir(), 'whisper-build-test-'));
    try {
        const bin = join(root, 'mock-bin');
        const cuda = join(root, 'cuda');
        const model = join(root, 'whisper.cpp/models/ggml-base.bin');
        const sharedBase = join(root, 'apps/shared/models/whisper/ggml-base.bin');
        const log = join(root, 'commands.log');
        for (const directory of [bin, join(cuda, 'bin'), join(root, 'whisper.cpp/models')]) {
            mkdirSync(directory, { recursive: true });
        }
        if (setup.legacyModel !== false) writeFileSync(model, 'original model');
        mkdirSync(join(root, 'apps/shared/models/whisper'), { recursive: true });
        if (setup.sharedModel !== false) writeFileSync(sharedBase, 'shared base model');
        writeFileSync(join(root, 'board'), failure === 'board' ? 'Jetson Xavier' : 'NVIDIA Jetson Orin Nano Super\0');
        writeFileSync(log, '');
        writeFileSync(join(bin, 'ssh'), `#!${process.execPath}
const { readFileSync } = require('node:fs');
const { spawnSync } = require('node:child_process');
const input = readFileSync(0, 'utf8')
    .replaceAll('/proc/device-tree/model', process.env.HOME + '/board')
    .replaceAll('/usr/local/cuda', process.env.HOME + '/cuda');
const result = spawnSync('bash', ['-c', process.argv.at(-1)], { input, stdio: ['pipe', 'inherit', 'inherit'] });
process.exit(result.status ?? 1);
`, { mode: 0o755 });
        const mock = join(bin, 'mock');
        writeFileSync(mock, `#!/usr/bin/env bash
set -euo pipefail
tool=\${0##*/}
printf '%s %s\\n' "$tool" "$*" >> "$TEST_LOG"
case "$tool" in
    uname)
        if [[ $1 == -s ]]; then echo Linux; else echo aarch64; fi ;;
    nvcc) echo 'CUDA 13.2' ;;
    curl)
        while [[ $1 != --output ]]; do shift; done
        printf 'downloaded base model' > "$2"
        [[ $TEST_FAILURE != download ]] || exit 22
        if [[ $TEST_FAILURE == empty-download ]]; then : > "$2"; fi ;;
    git)
        if [[ $1 == clone ]]; then
            [[ $TEST_FAILURE != clone ]] || exit 1
            mkdir -p "\${@: -1}/samples"
        elif [[ $1 == init ]]; then
            mkdir -p "\${@: -1}/samples"
            echo 'sample' > "\${@: -1}/samples/jfk.wav"
        elif [[ $* == *rev-parse* && $TEST_SHARED == yes ]]; then
            echo 0c7be696371069e0baa2515ed7aa08d5a5473aa6
        else
            echo 0123456789abcdef
        fi ;;
    cmake)
        case "$1" in
            -S)
                [[ $TEST_FAILURE != configure ]] || exit 1
                build=$4
                mkdir -p "$build/bin"
                printf 'GGML_CUDA:BOOL=ON\\nBUILD_SHARED_LIBS:BOOL=ON\\nCMAKE_CUDA_ARCHITECTURES:STRING=87\\n' > "$build/CMakeCache.txt"
                if [[ $TEST_FAILURE == cache ]]; then
                    printf 'GGML_CUDA:BOOL=OFF\\n' > "$build/CMakeCache.txt"
                fi
                ln -s "$TEST_MOCK" "$build/bin/whisper-cli" ;;
            --build) [[ $TEST_FAILURE != build ]] ;;
            --install)
                [[ $TEST_FAILURE != install ]] || exit 1
                prefix=\${@: -1}
                for file in include/whisper.h include/ggml.h lib/cmake/whisper/whisper-config.cmake lib/cmake/ggml/ggml-config.cmake lib/libwhisper.so lib/libggml.so lib/libggml-base.so lib/libggml-cpu.so lib/libggml-cuda.so; do
                    mkdir -p "$prefix/\${file%/*}"
                    echo 'installed' > "$prefix/$file"
                done
                mkdir -p "$prefix/bin"
                ln -s "$TEST_MOCK" "$prefix/bin/whisper-cli" ;;
        esac ;;
    ctest) [[ $TEST_FAILURE != tests ]] ;;
    ldd)
        if [[ $TEST_FAILURE == linkage ]]; then echo 'libwhisper.so => not found'; else echo 'resolved'; fi ;;
    whisper-cli)
        [[ $TEST_FAILURE != inference ]] || exit 1
        if [[ $TEST_FAILURE != cpu ]]; then
            echo 'whisper_backend_init_gpu: using CUDA0 backend'
        fi
        if [[ $TEST_FAILURE == backend ]]; then
            echo 'whisper_backend_init_gpu: failed to initialize CUDA0 backend'
        fi
        if [[ $TEST_FAILURE == buffer ]]; then
            echo 'whisper_init_state: CUDA0 compute buffer size = 0.00 MB'
        else
            echo 'whisper_init_state: CUDA0 compute buffer size = 188.45 MB'
        fi
        if [[ $TEST_FAILURE == transcript ]]; then
            echo 'wrong transcript' > "\${@: -1}.txt"
        else
            echo 'And so, ask not what your country can do for you.' > "\${@: -1}.txt"
        fi ;;
esac
`, { mode: 0o755 });
        for (const tool of ['git', 'cmake', 'ctest', 'ninja', 'g++', 'uname', 'ldd', 'curl']) {
            symlinkSync(mock, join(bin, tool));
        }
        symlinkSync(mock, join(cuda, 'bin/nvcc'));
        if (setup.ready) {
            const prefix = join(root, 'apps/shared/whisper/0c7be696-cuda');
            for (const file of ['include/whisper.h', 'include/ggml.h',
                'lib/cmake/whisper/whisper-config.cmake', 'lib/cmake/ggml/ggml-config.cmake',
                'lib/libwhisper.so', 'lib/libggml.so', 'lib/libggml-base.so',
                'lib/libggml-cpu.so', 'lib/libggml-cuda.so', '.ready']) {
                mkdirSync(join(prefix, file, '..'), { recursive: true });
                writeFileSync(join(prefix, file), 'existing SDK');
            }
            mkdirSync(join(prefix, 'bin'));
            symlinkSync(mock, join(prefix, 'bin/whisper-cli'));
            mkdirSync(join(root, 'apps/shared/test-data/whisper'), { recursive: true });
            writeFileSync(join(root, 'apps/shared/test-data/whisper/jfk.wav'), 'sample');
        }
        const result = spawnSync('bash', [script, ...args], {
            encoding: 'utf8',
            timeout: 10000,
            env: {
                ...process.env,
                PATH: `${bin}:${process.env.PATH}`,
                HOME: root,
                JETSON_JOBS: '2',
                JETSON_MODEL: '',
                TEST_FAILURE: failure,
                TEST_LOG: log,
                TEST_MOCK: mock,
                TEST_SHARED: args.includes('--install-shared') ? 'yes' : 'no',
                ...overrides,
            },
        });
        assert.ifError(result.error);
        return {
            ...result,
            log: readFileSync(log, 'utf8'),
            model: existsSync(model) ? readFileSync(model, 'utf8') : '',
            sharedBase: existsSync(sharedBase) ? readFileSync(sharedBase, 'utf8') : '',
            modelFiles: readdirSync(join(root, 'apps/shared/models/whisper')),
            workspaces: readdirSync(root).filter(name => name.startsWith('whisper-jetson.')),
            ready: (() => {
                try {
                    return readFileSync(join(root, 'apps/shared/whisper/0c7be696-cuda/.ready'), 'utf8');
                } catch (error) {
                    if (error.code !== 'ENOENT') throw error;
                    return '';
                }
            })(),
        };
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

test('shallow pinned Orin CUDA build is verified without installation', () => {
    const result = runBuild();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.log, /git clone --depth 1 --branch v1\.9\.5 https:\/\/github\.com\/ggml-org\/whisper\.cpp\.git /);
    for (const flag of ['CMAKE_BUILD_TYPE=Release', 'BUILD_SHARED_LIBS=ON', 'GGML_NATIVE=ON', 'GGML_CUDA=ON', 'CMAKE_CUDA_ARCHITECTURES=87']) {
        assert.ok(result.log.includes(`-D${flag}`), result.log);
    }
    assert.match(result.log, /cmake --build .* --parallel 2/);
    assert.match(result.log, /ctest .*--no-tests=error -R \^\(test-common-utf8\|test-whisper-buffer-loader\)\$/);
    assert.ok(result.log.indexOf('ctest ') < result.log.indexOf('whisper-cli '));
    assert.doesNotMatch(result.log, /--install|systemctl|sudo/);
    assert.equal(result.model, 'original model');
    assert.equal(result.workspaces.length, 1);
    assert.match(result.stdout, /Verified v1\.9\.5/);
});

for (const failure of ['clone', 'configure', 'build', 'cache', 'tests', 'inference', 'cpu', 'buffer', 'backend', 'transcript']) {
    test(`${failure} failure is reported and retains diagnostics`, () => {
        const result = runBuild(failure);
        assert.notEqual(result.status, 0);
        assert.doesNotMatch(result.stdout, /Verified v1\.9\.5/);
        assert.match(result.stderr, /diagnostic files retained/);
        const reached = {
            clone: /git clone /,
            configure: /cmake -S /,
            build: /cmake --build /,
            cache: /Shared libraries were not enabled|CUDA was not enabled/,
            tests: /ctest /,
            inference: /whisper-cli /,
            cpu: /did not select a CUDA backend/,
            buffer: /No nonzero CUDA compute buffer/,
            backend: /Backend initialization failed/,
            transcript: /JFK transcription did not match/,
        };
        assert.match(result.log + result.stderr, reached[failure]);
        assert.equal(result.workspaces.length, 1);
        assert.equal(result.model, 'original model');
        assert.doesNotMatch(result.log, /--install|systemctl|sudo/);
    });
}

for (const [name, failure, overrides, args] of [
    ['wrong board', 'board', {}, ['test@jetson']],
    ['invalid jobs', '', { JETSON_JOBS: '0' }, ['test@jetson']],
    ['missing model', '', { JETSON_MODEL: '/does/not/exist.bin' }, ['test@jetson']],
    ['missing host', '', {}, []],
    ['option as host', '', {}, ['-bad']],
]) {
    test(`${name} is rejected before cloning`, () => {
        const result = runBuild(failure, overrides, args);
        assert.notEqual(result.status, 0);
        assert.doesNotMatch(result.log, /git clone/);
        assert.deepEqual(result.workspaces, []);
    });
}

test('job count override is passed to the remote build', () => {
    const result = runBuild('', { JETSON_JOBS: '1' });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.log, /cmake --build .* --parallel 1/);
});

test('shared mode fetches the exact commit and publishes only after installed SDK verification', () => {
    const result = runBuild('', {}, ['test@jetson', '--install-shared']);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.log, /git -C .* fetch --depth 1 origin 0c7be696371069e0baa2515ed7aa08d5a5473aa6/);
    assert.match(result.log, /cmake --install .* --prefix .*\/apps\/shared\/whisper\/0c7be696-cuda/);
    assert.match(result.log, /installed-jfk/);
    assert.match(result.ready, /0c7be696/);
    assert.equal(result.model, 'original model');
    assert.doesNotMatch(result.log, /systemctl|sudo/);
});

test('default verification uses shared base without the legacy checkout', () => {
    const result = runBuild('', {}, ['test@jetson'], { legacyModel: false });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.log, /whisper-cli -m .*\/apps\/shared\/models\/whisper\/ggml-base\.bin /);
    assert.doesNotMatch(result.log, /curl /);
    assert.equal(result.sharedBase, 'shared base model');
});

test('shared provisioning downloads missing base before compiling', () => {
    const result = runBuild('', {}, ['test@jetson', '--install-shared'], { legacyModel: false, sharedModel: false });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.sharedBase, 'downloaded base model');
    assert.deepEqual(result.modelFiles, ['ggml-base.bin']);
    assert.match(result.log, /https:\/\/huggingface\.co\/ggerganov\/whisper\.cpp\/resolve\/main\/ggml-base\.bin/);
    assert.ok(result.log.indexOf('curl ') < result.log.indexOf('cmake -S '));
    assert.match(result.ready, /0c7be696/);
});

test('shared provisioning reuses legacy base without downloading', () => {
    const result = runBuild('', {}, ['test@jetson', '--install-shared'], { sharedModel: false });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.sharedBase, 'original model');
    assert.doesNotMatch(result.log, /curl /);
});

test('ready SDK repairs missing base without recompilation', () => {
    const result = runBuild('', {}, ['test@jetson', '--install-shared'], {
        legacyModel: false, sharedModel: false, ready: true,
    });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.sharedBase, 'downloaded base model');
    assert.doesNotMatch(result.log, /git |cmake /);
    assert.match(result.stdout, /skipping build/);
});

for (const failure of ['download', 'empty-download']) {
    test(`${failure} never publishes a partial base model or starts compilation`, () => {
        const result = runBuild(failure, {}, ['test@jetson', '--install-shared'], {
            legacyModel: false, sharedModel: false,
        });
        assert.notEqual(result.status, 0);
        assert.equal(result.sharedBase, '');
        assert.deepEqual(result.modelFiles, []);
        assert.equal(result.ready, '');
        assert.doesNotMatch(result.log, /git |cmake /);
    });
}

for (const failure of ['install', 'linkage', 'cpu', 'buffer', 'transcript']) {
    test(`shared ${failure} failure never publishes readiness`, () => {
        const result = runBuild(failure, {}, ['test@jetson', '--install-shared']);
        assert.notEqual(result.status, 0);
        assert.equal(result.ready, '');
        assert.doesNotMatch(result.log, /systemctl|sudo/);
    });
}
