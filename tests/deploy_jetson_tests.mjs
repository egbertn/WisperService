import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const deploy = fileURLToPath(new URL('../scripts/deploy-jetson.sh', import.meta.url));

function runDeployment(failure = '', extraArguments = []) {
    const root = mkdtempSync(join(tmpdir(), 'whisper-deploy-test-'));
    try {
        const prefix = join(root, 'apps/whisper-service');
        const repository = join(prefix, 'source');
        const bin = join(root, 'mock-bin');
        const log = join(root, 'commands.log');
        for (const directory of [
            repository, join(prefix, 'bin'),
            join(prefix, 'deps/whisper'), join(prefix, 'deps/ffmpeg/lib/pkgconfig'), bin,
        ]) {
            mkdirSync(directory, { recursive: true });
        }
        writeFileSync(join(repository, 'original-source.txt'), 'preserve this source');
        const executable = join(prefix, 'bin/whisper_transcribe');
        writeFileSync(executable, 'old executable', { mode: 0o755 });
        const shared = join(root, 'apps/shared');
        const ffmpeg = join(shared, 'ffmpeg/8.0.1-nvidia1-l4t39.2.1-rtsp-local1');
        const whisper = join(shared, 'whisper/0c7be696-cuda');
        for (const [base, files] of [
            [ffmpeg, ['.ready', 'lib/libavformat.so', 'lib/libavcodec.so', 'lib/libavutil.so', 'lib/libswscale.so', 'lib/libswresample.so']],
            [whisper, ['.ready', 'include/whisper.h', 'include/ggml.h', 'lib/cmake/whisper/whisper-config.cmake',
                'lib/cmake/ggml/ggml-config.cmake', 'lib/libwhisper.so', 'lib/libggml.so', 'lib/libggml-base.so',
                'lib/libggml-cpu.so', 'lib/libggml-cuda.so']],
            [shared, ['models/whisper/ggml-base.bin', 'test-data/whisper/jfk.wav']],
        ]) {
            for (const file of files) {
                const path = join(base, file);
                mkdirSync(join(path, '..'), { recursive: true });
                writeFileSync(path, 'existing dependency');
            }
        }
        writeFileSync(log, '');
        writeFileSync(join(bin, 'ssh'), '#!/usr/bin/env bash\nexec bash -c "${@: -1}"\n', { mode: 0o755 });
        const mock = join(bin, 'mock');
        writeFileSync(mock, `#!/usr/bin/env bash
set -euo pipefail
tool=\${0##*/}
printf '%s %s\\n' "$tool" "$*" >> "$TEST_LOG"
case "$tool" in
    git)
        case "$1" in
            init)
                [[ $TEST_FAILURE != init ]] || exit 1
                mkdir -p "\${@: -1}/.git" ;;
            fetch) [[ $TEST_FAILURE != fetch ]] ;;
            checkout) [[ $TEST_FAILURE != checkout ]] ;;
            rev-parse)
                case "$2" in
                    --verify) echo 0123456789abcdef ;;
                esac ;;
        esac ;;
    cmake)
        case "$1" in
            -S) [[ $TEST_FAILURE != configure ]] ;;
            --build) [[ $TEST_FAILURE != build ]] ;;
            --install)
                [[ $TEST_FAILURE != install ]] || exit 1
                mkdir -p "$DESTDIR$TEST_PREFIX/bin"
                printf 'new executable' > "$DESTDIR$TEST_PREFIX/bin/whisper_transcribe" ;;
        esac ;;
    ctest)
        [[ " $* " == *" --no-tests=error "* ]]
        [[ $TEST_FAILURE != tests && $TEST_FAILURE != no-tests ]] ;;
    pkg-config) echo "$TEST_FFMPEG" ;;
    ffmpeg)
        case "\${@: -1}" in
            -protocols)
                printf 'Input:\\nfile\\npipe\\nhttp\\ntcp\\nudp\\nrtp\\nOutput:\\nfile\\npipe\\n' ;;
            -demuxers)
                for format in rtsp rtp sdp mpegts h264 hevc mov,mp4,m4a,3gp,3g2,mj2 matroska,webm wav mp3 aac ogg flac aiff asf rm; do
                    echo " D $format Mock demuxer"
                done ;;
            -muxers)
                for format in mp4 mov ipod wav mp3 adts matroska webm ogg flac null; do
                    echo " E $format Mock muxer"
                done ;;
            *) echo 'h264_nvv4l2dec hevc_nvv4l2dec libx264 libx265 aac' ;;
        esac ;;
    ldd)
        if [[ $TEST_FAILURE == dependency ]]; then echo 'libwhisper.so.1 => not found'; exit 0; fi
        for library in libavformat libavcodec libavutil libswscale libswresample; do
            echo "$library.so.1 => $TEST_FFMPEG/lib/$library.so.1 (0x1234)"
        done
        for library in libwhisper libggml libggml-base libggml-cpu libggml-cuda; do
            path="$TEST_WHISPER/lib"
            if [[ $TEST_FAILURE == wrong-path && $1 == *stage* ]]; then path=/wrong/lib; fi
            echo "$library.so.1 => $path/$library.so.1 (0x1234)"
        done ;;
    curl)
        [[ $TEST_FAILURE != health ]] || exit 22
        echo '{"status":"ready"}' ;;
    sudo)
        shift
        case "$1" in
            install) exit 0 ;;
            cat)
                for library in libavformat libavcodec libavutil libswresample; do
                    echo "0-1 r-xp 00000000 00:00 1 $TEST_FFMPEG/lib/$library.so.1"
                done
                for library in libwhisper libggml libggml-base libggml-cpu libggml-cuda; do
                    echo "0-1 r-xp 00000000 00:00 1 $TEST_WHISPER/lib/$library.so.1"
                done
                if [[ $TEST_FAILURE == runtime-path ]]; then
                    echo '0-1 r-xp 00000000 00:00 1 /old/private/libggml.so.0'
                fi
                exit 0 ;;
        esac
        exec "$@" ;;
    systemctl)
        if [[ $1 == show ]]; then echo 1234; fi
        if [[ $1 == restart && $TEST_FAILURE == restart ]]; then exit 1; fi ;;
esac
`, { mode: 0o755 });
        for (const tool of ['git', 'cmake', 'ctest', 'sudo', 'systemctl', 'ninja', 'pkg-config', 'ldd', 'curl']) {
            symlinkSync(mock, join(bin, tool));
        }
        for (const [base, tool] of [[ffmpeg, 'ffmpeg'], [ffmpeg, 'ffprobe'], [whisper, 'whisper-cli']]) {
            mkdirSync(join(base, 'bin'), { recursive: true });
            symlinkSync(mock, join(base, 'bin', tool));
        }
        const result = spawnSync('bash', [
            deploy, 'test@jetson', ...extraArguments,
        ], {
            encoding: 'utf8',
            timeout: 10000,
            env: {
                ...process.env,
                PATH: `${bin}:${process.env.PATH}`,
                HOME: root,
                JETSON_JOBS: '',
                TEST_PREFIX: prefix,
                TEST_LOG: log,
                TEST_FAILURE: failure,
                TEST_FFMPEG: ffmpeg,
                TEST_WHISPER: whisper,
                TMPDIR: root,
            },
        });
        assert.ifError(result.error);
        return {
            ...result,
            log: readFileSync(log, 'utf8'),
            executable: readFileSync(executable, 'utf8'),
            originalSource: readFileSync(join(repository, 'original-source.txt'), 'utf8'),
            temporaryDirectories: readdirSync(root).filter(name => name.startsWith('whisper-deploy.')),
        };
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

test('successful tests precede installation and service restart', () => {
    const result = runDeployment();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.log, /cmake --build .* --parallel 6/);
    assert.equal(result.executable, 'new executable');
    const tests = result.log.indexOf('ctest ');
    const install = result.log.indexOf('cmake --install ');
    const restart = result.log.indexOf('systemctl restart ');
    assert.ok(tests >= 0 && tests < install && install < restart, result.log);
    assert.match(result.log, /systemctl is-active --quiet whisper_transcribe\.service/);
    assert.match(result.stdout, /is active/);
    assert.equal(result.originalSource, 'preserve this source');
    assert.deepEqual(result.temporaryDirectories, []);
    assert.match(result.log, /git remote add origin https:\/\/github.com\/egbertn\/WisperService.git/);
    assert.match(result.log, /git fetch --no-tags -- origin main/);
    assert.match(result.log, /cmake -S .*\/whisper-deploy\.[^/]+\/source -B .*\/whisper-deploy\.[^/]+\/build/);
    assert.match(result.log, /-DWHISPER_SERVICE_WHISPER_ROOT=.*\/apps\/shared\/whisper\/0c7be696-cuda/);
    assert.match(result.log, /-DWHISPER_SERVICE_FFMPEG_ROOT=.*\/apps\/shared\/ffmpeg\/8\.0\.1-nvidia1-l4t39\.2\.1-rtsp-local1/);
    assert.match(result.log, /20-shared-libraries\.conf/);
    assert.match(result.log, /curl .*http:\/\/jetson:8007\/health/);
});

test('installation uses the fixed path under the remote home directory', () => {
    const result = runDeployment();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.executable, 'new executable');
    assert.match(result.stdout, /Install prefix: .*\/apps\/whisper-service\n/);
    assert.deepEqual(result.temporaryDirectories, []);
});

test('custom installation paths and revisions are rejected', () => {
    const result = runDeployment('', ['/opt/another-service', 'another-branch']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Usage:/);
    assert.equal(result.log, '');
    assert.equal(result.executable, 'old executable');
});

for (const failure of ['init', 'fetch', 'checkout', 'configure', 'build', 'tests', 'no-tests', 'install']) {
    test(`${failure} failure preserves the installed executable and does not restart`, () => {
        const result = runDeployment(failure);
        assert.notEqual(result.status, 0);
        assert.equal(result.executable, 'old executable');
        assert.equal(result.originalSource, 'preserve this source');
        assert.deepEqual(result.temporaryDirectories, []);
        assert.doesNotMatch(result.log, /systemctl restart /);
        const reached = {
            init: /git init /,
            fetch: /git fetch /,
            checkout: /git checkout /,
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
    assert.deepEqual(result.temporaryDirectories, []);
});

for (const failure of ['dependency', 'wrong-path']) {
    test(`${failure} failure preserves the binary and does not restart`, () => {
        const result = runDeployment(failure);
        assert.notEqual(result.status, 0);
        assert.equal(result.executable, 'old executable');
        assert.doesNotMatch(result.log, /systemctl restart |systemctl daemon-reload/);
    });
}

for (const failure of ['health', 'runtime-path']) {
    test(`${failure} failure after restart is not presented as success`, () => {
        const result = runDeployment(failure);
        assert.notEqual(result.status, 0);
        assert.match(result.log, /systemctl restart /);
        assert.doesNotMatch(result.stdout, /is active/);
    });
}
