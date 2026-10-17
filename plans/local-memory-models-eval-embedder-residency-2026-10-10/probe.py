"""Whisper probe + embedder load generator for the residency measurement.

Run under whisper-writer's venv (same faster-whisper/CTranslate2 build as the resident
process). Writes JSON-lines events to --out; GPU sampling is done by separate
nvidia-smi processes started by run_phase.sh.
"""
import argparse
import difflib
import json
import random
import string
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

HERE = Path(__file__).parent
REFERENCE = ('And so my fellow Americans, ask not what your country can do for you, '
             'ask what you can do for your country.')


def words(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()


def match_ratio(text):
    return difflib.SequenceMatcher(None, words(REFERENCE), words(text)).ratio()


class Log:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, event, **data):
        rec = {'t': time.time(), 'event': event, **data}
        with self.lock, open(self.path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)


def transcribe(model, audio):
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio=audio, language='en', initial_prompt=None,
        condition_on_previous_text=False, temperature=0.0, vad_filter=False,
    )
    text = ''.join(s.text for s in segments).strip()
    return time.perf_counter() - started, text


def whisper_battery(model, log, phase, clip, clip30, n_short=5, n_long=3):
    for i in range(n_short):
        secs, text = transcribe(model, clip)
        log('transcribe', phase=phase, clip='jfk11s', i=i, secs=round(secs, 4),
            match=round(match_ratio(text), 4), text=text)
    for i in range(n_long):
        secs, text = transcribe(model, clip30)
        log('transcribe', phase=phase, clip='jfk30s', i=i, secs=round(secs, 4), text=text[:200])


class EmbedLoad:
    """Background embedding load: Poisson single-text queries or batched backfill."""

    def __init__(self, url, model, log, phase):
        self.url, self.model, self.log, self.phase = url, model, log, phase
        self.stop = threading.Event()
        self.threads = []
        self.lock = threading.Lock()
        self.ok_vectors = 0
        self.ok_requests = 0
        self.errors = 0
        self.latencies = []

    def _post(self, inputs, truncate=None):
        body = {'model': self.model, 'input': inputs}
        if truncate:
            body['truncate_prompt_tokens'] = truncate
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
            n = len(payload['data'])
            with self.lock:
                self.ok_vectors += n
                self.ok_requests += 1
                self.latencies.append(time.perf_counter() - started)
        except Exception as exc:  # recorded, never swallowed silently
            with self.lock:
                self.errors += 1
            self.log('embed_error', phase=self.phase, error=repr(exc)[:300])

    def start_queries(self, texts, rate):
        def run(_worker):
            rng = random.Random(1)
            while not self.stop.is_set():
                self._post([rng.choice(texts)])
                self.stop.wait(rng.expovariate(rate))
        self._spawn(run, 1)

    def start_backfill(self, texts, batch, concurrency, truncate=None):
        def run(offset):
            i = offset * batch
            while not self.stop.is_set():
                chunk = [texts[(i + k) % len(texts)] for k in range(batch)]
                i += batch * concurrency
                self._post(chunk, truncate)
        self._spawn(run, concurrency)

    def start_paced(self, texts, batch, interval):
        def run(_worker):
            i = 0
            while not self.stop.is_set():
                tick = time.perf_counter()
                self._post([texts[(i + k) % len(texts)] for k in range(batch)])
                i += batch
                self.stop.wait(max(0.0, interval - (time.perf_counter() - tick)))
        self._spawn(run, 1)

    def _spawn(self, fn, n):
        self.started = time.perf_counter()
        for k in range(n):
            th = threading.Thread(target=fn, args=(k,), daemon=True)
            th.start()
            self.threads.append(th)

    def finish(self):
        self.stop.set()
        for th in self.threads:
            th.join(timeout=180)
        elapsed = time.perf_counter() - self.started
        lat = sorted(self.latencies)
        summary = {
            'phase': self.phase, 'elapsed_s': round(elapsed, 2),
            'ok_requests': self.ok_requests, 'ok_vectors': self.ok_vectors,
            'errors': self.errors,
            'vectors_per_s': round(self.ok_vectors / elapsed, 1) if elapsed else None,
            'req_latency_p50_ms': round(lat[len(lat) // 2] * 1000, 1) if lat else None,
            'req_latency_p95_ms': round(lat[int(len(lat) * 0.95)] * 1000, 1) if lat else None,
        }
        self.log('load_summary', **summary)
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['baseline', 'loaded', 'loaded2', 'ab'], required=True)
    ap.add_argument('--embed-url', default='http://127.0.0.1:8415/v1/embeddings')
    ap.add_argument('--embed-model', default='granite-embedding-english-r2')
    ap.add_argument('--query-rate', type=float, default=5.0)
    ap.add_argument('--stress-query-rate', type=float, default=50.0)
    ap.add_argument('--query-prefix', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    log = Log(args.out)

    clip = decode_audio(str(HERE / 'audio' / 'jfk.flac'))
    clip30 = np.concatenate([clip, clip, clip])[:30 * 16000]
    t0 = time.perf_counter()
    model = WhisperModel('large-v3', device='cuda', compute_type='float16')
    log('whisper_loaded', secs=round(time.perf_counter() - t0, 2))
    secs, text = transcribe(model, clip)
    log('warmup', secs=round(secs, 4), match=round(match_ratio(text), 4))

    if args.phase == 'baseline':
        log('phase_start', phase='no_embedder')
        whisper_battery(model, log, 'no_embedder', clip, clip30)
        log('phase_end', phase='no_embedder')
        return

    texts = json.loads((HERE / 'data' / 'mem0_texts.json').read_text())
    if args.phase == 'ab':
        return interleaved_ab(model, log, args, texts, clip)
    if args.phase == 'loaded2':
        return second_pass(model, log, args, texts, clip, clip30)
    queries = [args.query_prefix + ' '.join(t.split()[:25]) for t in texts]
    long_texts = []
    joined = ' '.join(texts)
    for k in range(64):
        start = (k * 40000) % max(1, len(joined) - 40000)
        long_texts.append(joined[start:start + 40000])

    log('phase_start', phase='embedder_idle')
    whisper_battery(model, log, 'embedder_idle', clip, clip30)
    log('phase_end', phase='embedder_idle')

    for phase, rate in (('query_rate', args.query_rate), ('query_stress', args.stress_query_rate)):
        log('phase_start', phase=phase, rate=rate)
        load = EmbedLoad(args.embed_url, args.embed_model, log, phase)
        load.start_queries(queries, rate)
        time.sleep(3)
        whisper_battery(model, log, phase, clip, clip30)
        load.finish()
        log('phase_end', phase=phase)

    log('phase_start', phase='backfill_real', batch=64, concurrency=4)
    load = EmbedLoad(args.embed_url, args.embed_model, log, 'backfill_real')
    load.start_backfill(texts, 64, 4)
    time.sleep(5)
    whisper_battery(model, log, 'backfill_real', clip, clip30)
    load.finish()
    log('phase_end', phase='backfill_real')

    log('phase_start', phase='backfill_maxlen', batch=64, concurrency=4, truncate=8192)
    load = EmbedLoad(args.embed_url, args.embed_model, log, 'backfill_maxlen')
    load.start_backfill(long_texts, 64, 4, truncate=8000)
    time.sleep(5)
    whisper_battery(model, log, 'backfill_maxlen', clip, clip30, n_short=3, n_long=3)
    load.finish()
    log('phase_end', phase='backfill_maxlen')

    log('phase_start', phase='cooldown')
    time.sleep(20)
    log('phase_end', phase='cooldown')


def interleaved_ab(model, log, args, texts, clip, rounds=10):
    queries = [args.query_prefix + ' '.join(t.split()[:25]) for t in texts]
    conditions = [
        ('ab_idle', None),
        ('ab_query_3_6', lambda load: load.start_queries(queries, 5.0)),
        ('ab_query_12', lambda load: load.start_queries(queries, 16.0)),
        ('ab_write_80', lambda load: load.start_paced(texts, 16, 0.2)),
    ]
    for r in range(rounds):
        for name, starter in conditions:
            load = None
            if starter:
                load = EmbedLoad(args.embed_url, args.embed_model, log, f'{name}_r{r}')
                starter(load)
                time.sleep(1.5)
            secs, text = transcribe(model, clip)
            log('ab', condition=name, round=r, secs=round(secs, 4), match=round(match_ratio(text), 4))
            if load:
                load.finish()


def second_pass(model, log, args, texts, clip, clip30):
    log('phase_start', phase='embedder_idle2')
    whisper_battery(model, log, 'embedder_idle2', clip, clip30)
    log('phase_end', phase='embedder_idle2')
    plans = [
        ('write_burst_80vps', lambda load: load.start_paced(texts, 16, 0.2)),
        ('backfill_c1_b16', lambda load: load.start_backfill(texts, 16, 1)),
        ('backfill_c1_b64', lambda load: load.start_backfill(texts, 64, 1)),
        ('backfill_real_repeat', lambda load: load.start_backfill(texts, 64, 4)),
    ]
    for phase, starter in plans:
        log('phase_start', phase=phase)
        load = EmbedLoad(args.embed_url, args.embed_model, log, phase)
        starter(load)
        time.sleep(4)
        whisper_battery(model, log, phase, clip, clip30)
        load.finish()
        log('phase_end', phase=phase)
    log('phase_start', phase='cooldown2')
    time.sleep(15)
    log('phase_end', phase='cooldown2')


if __name__ == '__main__':
    main()
