import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('../scripts/build-jetson-ffmpeg.sh', import.meta.url));

function runBuild(failure = '', existing = false) {
    const root = mkdtempSync(join(tmpdir(), 'ffmpeg-build-test-'));
    try {
        const bin = join(root, 'mock-bin');
        const prefix = join(root, 'apps/shared/ffmpeg/8.0.1-nvidia1-l4t39.2.1-rtsp-local1');
        const log = join(root, 'commands.log');
        mkdirSync(bin);
        writeFileSync(log, '');
        writeFileSync(join(bin, 'ssh'), '#!/usr/bin/env bash\nexec bash -c "${@: -1}"\n', { mode: 0o755 });
        const mock = join(bin, 'mock');
        writeFileSync(mock, `#!/usr/bin/env bash
set -euo pipefail
tool=\${0##*/}
printf '%s %s\\n' "$tool" "$*" >> "$TEST_LOG"
case "$tool" in
    uname) echo aarch64 ;;
    dpkg-query) if [[ $TEST_FAILURE == platform ]]; then echo 38.0; else echo 39.2.1; fi ;;
    curl) [[ $TEST_FAILURE != download ]] ;;
    sha256sum) cat >/dev/null; [[ $TEST_FAILURE != checksum ]] ;;
    tar)
        mkdir -p "$4"
        ln -s "$TEST_MOCK" "$4/configure" ;;
    configure) [[ $TEST_FAILURE != configure ]] ;;
    make)
        if [[ $1 == install ]]; then
            [[ $TEST_FAILURE != install ]] || exit 1
            mkdir -p "$TEST_PREFIX/bin" "$TEST_PREFIX/lib/pkgconfig"
            ln -s "$TEST_MOCK" "$TEST_PREFIX/bin/ffmpeg"
            ln -s "$TEST_MOCK" "$TEST_PREFIX/bin/ffprobe"
            for library in libavformat libavcodec libavutil libswscale libswresample; do
                echo library > "$TEST_PREFIX/lib/$library.so"
            done
        else
            [[ $TEST_FAILURE != build ]]
        fi ;;
    pkg-config)
        if [[ $TEST_FAILURE == pkg-config ]]; then echo /wrong/prefix; else echo "$TEST_PREFIX"; fi ;;
    ldd)
        if [[ $TEST_FAILURE == linkage ]]; then
            echo 'libavcodec.so.1 => not found'
        else
            echo "libavcodec.so.1 => $TEST_PREFIX/lib/libavcodec.so.1 (0x1234)"
        fi ;;
    ffmpeg)
        case "\${@: -1}" in
            -decoders)
                if [[ $TEST_FAILURE != decoders ]]; then echo 'h264_nvv4l2dec hevc_nvv4l2dec'; fi ;;
            -encoders)
                if [[ $TEST_FAILURE != encoders ]]; then echo 'libx264 libx265'; fi
                if [[ $TEST_FAILURE != aac ]]; then echo aac; fi ;;
            -protocols)
                printf 'Supported file protocols:\\nInput:\\n'
                for protocol in file pipe http tcp udp rtp; do
                    if [[ $TEST_FAILURE != protocols || $protocol != tcp ]]; then echo "  $protocol"; fi
                done
                printf 'Output:\\n  file\\n  pipe\\n'
                if [[ $TEST_FAILURE == extra-protocol ]]; then echo '  rtmp'; fi ;;
            -demuxers)
                printf 'Formats:\\n D.. = Demuxing supported\\n .E. = Muxing supported\\n ..d = Is a device\\n ---\\n'
                printf ' D d fbdev Linux framebuffer\\n D d lavfi Libavfilter virtual input device\\n D d video4linux2,v4l2 Video4Linux2 device grab\\n'
                for format in rtsp rtp sdp mpegts h264 hevc mov,mp4,m4a,3gp,3g2,mj2 matroska,webm wav mp3 aac ogg flac aiff asf rm; do
                    if [[ $TEST_FAILURE != demuxers || $format != wav ]]; then echo " D  $format Mock demuxer"; fi
                done
                if [[ $TEST_FAILURE == extra-demuxer ]]; then echo ' D  hls Mock demuxer'; fi ;;
            -muxers)
                printf 'Formats:\\n D.. = Demuxing supported\\n .E. = Muxing supported\\n ..d = Is a device\\n ---\\n'
                printf '  Ed fbdev Linux framebuffer\\n  Ed oss OSS playback\\n  Ed video4linux2,v4l2 Video4Linux2 output device\\n'
                for format in mp4 mov ipod wav mp3 adts matroska webm ogg flac null; do
                    if [[ $TEST_FAILURE != muxers || $format != mp4 ]]; then echo "  E $format Mock muxer"; fi
                done
                if [[ $TEST_FAILURE == extra-muxer ]]; then echo '  E rtsp Mock muxer'; fi ;;
        esac ;;
esac
`, { mode: 0o755 });
        for (const tool of ['uname', 'dpkg-query', 'curl', 'sha256sum', 'tar', 'make', 'pkg-config', 'ldd']) {
            symlinkSync(mock, join(bin, tool));
        }
        if (existing) {
            mkdirSync(join(prefix, 'bin'), { recursive: true });
            mkdirSync(join(prefix, 'lib/pkgconfig'), { recursive: true });
            writeFileSync(join(prefix, '.ready'), 'verified');
            symlinkSync(mock, join(prefix, 'bin/ffmpeg'));
            symlinkSync(mock, join(prefix, 'bin/ffprobe'));
            for (const library of ['libavformat', 'libavcodec', 'libavutil', 'libswscale', 'libswresample']) {
                writeFileSync(join(prefix, 'lib', `${library}.so`), 'library');
            }
        }
        const result = spawnSync('bash', [script, 'test@jetson'], {
            encoding: 'utf8',
            timeout: 10000,
            env: {
                ...process.env, HOME: root, PATH: `${bin}:${process.env.PATH}`,
                JETSON_JOBS: '', TEST_PREFIX: prefix, TEST_MOCK: mock, TEST_LOG: log, TEST_FAILURE: failure,
            },
        });
        assert.ifError(result.error);
        return { ...result, log: readFileSync(log, 'utf8'), ready: existsSync(join(prefix, '.ready')) };
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

test('FFmpeg is verified and installed with the shared video/audio feature set', () => {
    const result = runBuild();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.ready, true);
    assert.match(result.log, /make -j6/);
    assert.match(result.log, /configure .*--enable-gpl .*--enable-libx264 --enable-libx265 .*--enable-nvv4l2dec/);
    assert.match(result.log, /--disable-autodetect/);
    assert.match(result.log, /--disable-protocols --enable-protocol=file,pipe,http,tcp,udp,rtp(?: |$)/);
    assert.match(result.log, /--disable-demuxers --enable-demuxer=rtsp,rtp,sdp,mpegts,h264,hevc,mov,matroska,wav,mp3,aac,ogg,flac,aiff,asf(?: |$)/);
    assert.match(result.log, /--disable-muxers --enable-muxer=mp4,mov,ipod,wav,mp3,adts,matroska,webm,ogg,flac,null(?: |$)/);
    assert.doesNotMatch(result.log, /--disable-network|--disable-swscale|systemctl|sudo/);
    assert.ok(result.log.indexOf('sha256sum ') < result.log.indexOf('configure '));
    assert.ok(result.log.indexOf('make install') < result.log.indexOf('ffmpeg -hide_banner -decoders'));
});

test('verified shared FFmpeg is checked and reused without recompilation', () => {
    const result = runBuild('', true);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /skipping build/);
    assert.doesNotMatch(result.log, /curl |make |configure /);
    assert.match(result.log, /pkg-config /);
});

test('broken existing installation fails rather than being silently reused or overwritten', () => {
    const result = runBuild('pkg-config', true);
    assert.notEqual(result.status, 0);
    assert.doesNotMatch(result.log, /make |configure /);
});

for (const failure of ['platform', 'download', 'checksum', 'configure', 'build', 'install', 'pkg-config', 'linkage', 'decoders', 'encoders', 'aac', 'protocols', 'extra-protocol', 'demuxers', 'extra-demuxer', 'muxers', 'extra-muxer']) {
    test(`FFmpeg ${failure} failure never publishes readiness`, () => {
        const result = runBuild(failure);
        assert.notEqual(result.status, 0);
        assert.equal(result.ready, false);
    });
}

for (const failure of ['protocols', 'extra-protocol', 'demuxers', 'extra-demuxer', 'muxers', 'extra-muxer']) {
    test(`existing FFmpeg with ${failure} mismatch is rejected without rebuilding`, () => {
        const result = runBuild(failure, true);
        assert.notEqual(result.status, 0);
        assert.doesNotMatch(result.log, /make |configure /);
    });
}

for (const [failure, message] of [
    ['demuxers', 'Missing demuxers component: wav'],
    ['extra-demuxer', 'Unexpected demuxers component: hls'],
    ['muxers', 'Missing muxers component: mp4'],
    ['extra-muxer', 'Unexpected muxers component: rtsp'],
]) {
    test(`format verification reports ${failure} despite device rows`, () => {
        const result = runBuild(failure);
        assert.notEqual(result.status, 0);
        assert.equal(result.ready, false);
        assert.ok(result.stderr.includes(`Error: ${message}`), result.stderr);
    });
}
