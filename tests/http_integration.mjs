import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const [base, fixtures] = process.argv.slice(2);
if (!base || !fixtures) {
    throw new Error('Usage: node tests/http_integration.mjs http://jetson:8007 /path/to/fixtures');
}
const apiKey = process.env.WHISPER_API_KEY;
if (!apiKey || apiKey.length < 32) {
    throw new Error('Set WHISPER_API_KEY to the service API key (at least 32 characters)');
}
const post = (body, query = '', contentType = 'application/octet-stream', key = apiKey) => {
    const headers = { 'Content-Type': contentType };
    if (key) {
        headers.Authorization = `Bearer ${key}`;
    }
    return fetch(`${base}/?${query}`, {
        method: 'POST', body, headers,
        signal: AbortSignal.timeout(120000),
    });
};

const health = await fetch(`${base}/health`);
assert.equal(health.status, 200);
assert.equal((await health.json()).status, 'ready');
console.log('PASS health');

for (const file of ['speech.wav', 'speech32.wav', 'speech-float.wav', 'speech-stereo.wav',
                    'speech.mp3', 'speech.m4a', 'speech.webm', 'speech.ogg', 'speech.flac']) {
    const response = await post(await readFile(join(fixtures, file)), 'lang=en');
    assert.equal(response.status, 200, `${file}: ${await response.clone().text()}`);
    assert.match(response.headers.get('content-type'), /application\/json/);
    const result = await response.json();
    assert.match(result.text.toLowerCase(), /country/);
    assert.equal(result.language, 'en');
    assert.equal(result.task, 'transcribe');
    assert.equal(result.audio.sample_rate, 16000);
    assert.equal(result.audio.channels, 1);
    assert.ok(Math.abs(result.audio.duration_seconds - 11) < 0.2);
    assert.ok(result.processing_seconds > 0);
    assert.ok(result.model.length > 0);
    assert.ok(result.segments.length > 0);
    for (const segment of result.segments) {
        assert.ok(segment.start >= 0 && segment.end >= segment.start);
        assert.ok(segment.end <= result.audio.duration_seconds);
        assert.ok(segment.no_speech_probability >= 0 && segment.no_speech_probability <= 1);
    }
    if (file === 'speech-stereo.wav') {
        assert.equal(result.audio.original_sample_rate, 48000);
        assert.equal(result.audio.original_channels, 2);
    }
    console.log(`PASS ${file}: ${result.audio.codec}, ${result.audio.duration_seconds}s`);
}

const speech = await readFile(join(fixtures, 'speech.wav'));
const unauthorized = await post(speech, '', 'application/octet-stream', null);
assert.equal(unauthorized.status, 401);
assert.equal((await unauthorized.json()).error.status, 401);
const invalidKey = await post(speech, '', 'application/octet-stream', `${apiKey}-invalid`);
assert.equal(invalidKey.status, 401);
assert.equal((await invalidKey.json()).error.status, 401);
console.log('PASS API key required');

const text = await post(speech, 'lang=en&format=text');
assert.equal(text.status, 200);
assert.match(text.headers.get('content-type'), /text\/plain/);
assert.match((await text.text()).toLowerCase(), /country/);
console.log('PASS legacy text response');

const auto = await post(speech);
assert.equal(auto.status, 200);
assert.equal((await auto.json()).language, 'en');
console.log('PASS automatic language detection');

const translation = await post(speech, 'lang=en&translate=true');
assert.equal(translation.status, 200);
assert.equal((await translation.json()).task, 'translate');
console.log('PASS translation task');

for (const [body, query, status, type] of [
    ['', '', 400],
    ['invalid media', '', 422],
    [speech, 'lang=invalid-language', 400],
    [speech, 'translate=yes', 400],
    [speech, 'format=xml', 400],
    ['--test\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n' +
     'Content-Type: audio/wav\r\n\r\ninvalid\r\n--test--\r\n', '', 415, 'multipart/form-data; boundary=test'],
    [Buffer.alloc(32 * 1024 * 1024 + 1), '', 413],
    [await readFile(join(fixtures, 'too-long.flac')), '', 413],
]) {
    const response = await post(body, query, type);
    assert.equal(response.status, status, `${query}: ${await response.clone().text()}`);
    const result = await response.json();
    assert.equal(result.error.status, status);
    assert.ok(result.error.message.length > 0);
    console.log(`PASS error ${status} ${query}`);
}

const parallel = await Promise.all(Array.from({ length: 8 }, () => post(speech, 'lang=en')));
assert.ok(parallel.some(response => response.status === 503), 'Queue must reject overload');
assert.ok(parallel.some(response => response.status === 200), 'Queue must process admitted requests');
for (const response of parallel) {
    assert.ok([200, 503].includes(response.status), `Unexpected concurrent status: ${response.status}`);
    await response.arrayBuffer();
}
assert.equal((await post(speech, 'lang=en')).status, 200, 'Service must recover after overload and errors');
console.log('PASS bounded concurrency and request recovery');
