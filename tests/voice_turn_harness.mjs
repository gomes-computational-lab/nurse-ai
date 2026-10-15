// Run the real component against deterministic microphone/clock/browser mocks.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { default: mount } = await import(pathToFileURL(process.argv[2]));

let now, volume, timers, events, tracks, contexts, clips, speech, cancelled;
function reset() {
    now = 0; volume = 0; timers = new Set(); events = []; tracks = [];
    contexts = []; clips = []; speech = null; cancelled = false;
}
Object.defineProperty(globalThis, 'performance', { value: { now: () => now } });
globalThis.setInterval = callback => { timers.add(callback); return callback; };
globalThis.clearInterval = callback => timers.delete(callback);
Object.defineProperty(globalThis, 'navigator', { value: { mediaDevices: {
    async getUserMedia() {
        const track = { stopped: false, stop() { this.stopped = true; } };
        tracks.push(track);
        return { getTracks: () => [track] };
    },
} } });
globalThis.AudioContext = class {
    constructor() { this.state = 'running'; contexts.push(this); }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createAnalyser() { return { fftSize: 2048, getFloatTimeDomainData(a) { a.fill(volume); } }; }
    async close() { this.state = 'closed'; }
};
globalThis.MediaRecorder = class {
    constructor() { this.state = 'inactive'; this.mimeType = 'audio/webm'; clips.push(this); }
    start() { this.state = 'recording'; }
    stop() {
        this.state = 'inactive';
        this.ondataavailable({ data: new Blob(['audio']) });
        this.onstop();
    }
};
globalThis.FileReader = class {
    readAsDataURL() { this.result = 'data:audio/webm;base64,YXVkaW8='; this.onload(); }
};
globalThis.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
globalThis.window = { speechSynthesis: {
    speak(utterance) { speech = utterance; },
    cancel() { cancelled = true; },
} };

async function start(reply = '') {
    const elements = { '#status': {}, '#enable': {} };
    const cleanup = mount({
        data: { reply, pause_ms: 700, threshold: 0.015 },
        parentElement: { querySelector: selector => elements[selector] },
        setStateValue: (key, payload) => events.push({ key, payload }),
    });
    await Promise.resolve();
    return cleanup;
}
function tick(duration, amplitude = 0) {
    volume = amplitude;
    for (let elapsed = 0; elapsed < duration; elapsed += 30) {
        now += 30;
        for (const timer of [...timers]) timer();
    }
}

reset();
let cleanup = await start();
tick(12000);
assert.equal(events.length, 0, 'Silence must not submit');
assert.equal(clips.length, 2, 'Idle capture restarts without submitting');
tick(90, 0.1); tick(750);
assert.equal(events.length, 0, 'Short noise must not submit');
tick(300, 0.1); tick(690);
assert.equal(events.length, 0, 'Short pause must not end speech');
tick(30);
assert.equal(events.length, 1, 'Finished utterance submits automatically');
assert.equal(events[0].payload.base64, 'YXVkaW8=');
assert.ok(tracks.every(t => t.stopped), 'Capture stops during response');
tick(3000, 0.1);
assert.equal(events.length, 1, 'Utterance is submitted once');
cleanup();

reset();
cleanup = await start('Patient reply');
assert.equal(tracks.length, 0, 'Do not capture the spoken reply');
speech.onend();
await Promise.resolve();
assert.equal(tracks.length, 1, 'Automatically listen after playback');
tick(300, 0.1);
cleanup();
tick(900);
assert.equal(events.length, 0, 'Ending must not submit late audio');
assert.ok(tracks.every(t => t.stopped));
assert.ok(contexts.every(c => c.state === 'closed'));
assert.ok(cancelled);

reset();
cleanup = await start();
tick(60030, 0.1);
assert.equal(events.length, 1, 'Long turns are bounded');
cleanup();
console.log('Browser pause, playback gating, idle/noise handling, and cleanup passed.');
