"""Worker pool: spawning, warming, supervision.

Workers are replaced when they die, overrun a timeout, or exceed the RSS
ceiling.
"""
from __future__ import annotations

import collections
import concurrent.futures as futures
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time

log = logging.getLogger("oracle.pool")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")


class Worker:
    """One Boltz shard pinned to a GPU, addressed over a newline JSON protocol."""

    def __init__(self, wid: int, gpu: int):
        self.wid, self.gpu = wid, gpu
        self.lock = threading.Lock()
        self.requests = 0
        self.proc: subprocess.Popen | None = None
        self.spawn()

    def spawn(self) -> None:
        # Shards sharing a Triton cache race when compiling the same kernel.
        triton = os.path.expanduser(f"~/.triton/cache/oracle-shard{self.wid}")
        env = dict(os.environ, WORKER_GPU=str(self.gpu), TRITON_CACHE_DIR=triton,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.messages: queue.Queue = queue.Queue()
        self.stderr: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._read_stdout, args=(self.proc.stdout, self.messages),
                         daemon=True).start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, args=(self.proc.stderr, self.stderr), daemon=True)
        self._stderr_reader.start()
        self._await_event("ready", timeout=120)
        self.requests = 0

    @staticmethod
    def _read_stdout(stream, messages: queue.Queue) -> None:
        # A thread, not select(): select() misses lines already in the read buffer.
        for line in stream:
            start = line.find('{"event"')  # progress output can precede it on the line
            if start < 0:
                continue
            try:
                messages.put(json.loads(line[start:]))
            except json.JSONDecodeError:
                continue
        messages.put(None)

    @staticmethod
    def _read_stderr(stream, tail: collections.deque) -> None:
        for line in stream:
            tail.append(line.rstrip())

    def tail(self) -> str:
        self._stderr_reader.join(timeout=2)
        return "\n".join(self.stderr)

    def _await_event(self, event: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        while (remaining := deadline - time.time()) > 0:
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty:
                break
            if message is None:
                raise RuntimeError(f"worker {self.wid} exited:\n{self.tail()}")
            if message.get("event") == event:
                return message
        raise TimeoutError(f"worker {self.wid}: no {event!r} within {timeout}s")

    def score(self, items: list[dict], timeout: float) -> dict:
        """items: [{"id", "sequence", "msa", "smiles"}] -- one batch for this shard."""
        self.proc.stdin.write(json.dumps({"items": items}) + "\n")
        self.proc.stdin.flush()
        message = self._await_event("result", timeout)
        self.requests += 1
        return message

    def rss_gb(self) -> float:
        try:
            with open(f"/proc/{self.proc.pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 1048576
        except OSError:
            pass
        return 0.0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)


class Pool:
    """Owns every shard and keeps the fleet healthy."""

    # Ubiquitin, single-sequence mode: a cold start needs no network.
    PROBE = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
             "KESTLHLVLRLRGG", "empty")
    PROBE_LIGAND = "CC(=O)Oc1ccccc1C(=O)O"

    def __init__(self, cfg):
        self.cfg = cfg
        # Breadth-first, so a batch smaller than the pool spreads one-per-card
        # rather than stacking on gpu0. Worth 2.5x on small batches.
        placement = [gpu for _ in range(cfg.shards_per_gpu) for gpu in cfg.gpus]

        def start(wid: int) -> Worker:
            time.sleep(wid * cfg.stagger_s)
            worker = Worker(wid, placement[wid])
            log.info("shard %d up on gpu%d", wid, placement[wid])
            return worker

        with futures.ThreadPoolExecutor(max_workers=len(placement)) as pool:
            self.workers: list[Worker] = list(pool.map(start, range(len(placement))))
        self._warm = self.PROBE
        self._stop = threading.Event()
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()

    def live(self) -> list[Worker]:
        return [worker for worker in self.workers if worker.alive()]

    def warm(self, sequence: str | None = None, msa: str | None = None) -> float:
        """Force the lazy checkpoint load so the first real request is fast.

        A given protein becomes the probe for later recycles too. A shard that
        will not warm is left to the supervisor."""
        if sequence is not None:
            self._warm = (sequence, msa)
        started = time.time()
        with futures.ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            list(pool.map(self._warm_one, self.workers))
        return time.time() - started

    def _warm_one(self, worker: Worker) -> None:
        # The lock _dispatch takes: without it a warm races a concurrent
        # recycle and blocks until timeout on a process already replaced.
        with worker.lock:
            try:
                worker.score(self._probe(), self.cfg.request_timeout_s)
            except Exception:  # noqa: BLE001 - one cold shard, not a failed round
                log.exception("shard %d failed to warm", worker.wid)
                worker.kill()  # a late reply would answer the next request

    def _probe(self) -> list[dict]:
        sequence, msa = self._warm
        return [{"id": "warm", "sequence": sequence, "msa": msa,
                 "smiles": self.PROBE_LIGAND}]

    def _supervise(self) -> None:
        while not self._stop.wait(30):
            for worker in self.workers:
                if not worker.alive():
                    self._replace(worker, "died")
                elif worker.rss_gb() > self.cfg.recycle_rss_gb:
                    self._replace(worker, "rss")

    def _replace(self, worker: Worker, reason: str) -> None:
        if not worker.lock.acquire(blocking=False):
            return  # busy; catch it on the next sweep
        try:
            log.warning("recycling shard %d on gpu%d (%s, rss=%.1f GB, %d requests)",
                        worker.wid, worker.gpu, reason, worker.rss_gb(), worker.requests)
            if reason == "died":
                log.warning("shard %d stderr:\n%s", worker.wid, worker.tail())
            worker.kill()
            worker.spawn()
            # Warm under the lock, so the ~60 s falls on one waiter rather
            # than on whichever request lands here next.
            worker.score(self._probe(), self.cfg.request_timeout_s)
        except Exception:
            log.exception("failed to recycle shard %d", worker.wid)
            worker.kill()
        finally:
            worker.lock.release()

    def health(self) -> dict:
        return {"alive": len(self.live()), "total": len(self.workers),
                "gpus": sorted({w.gpu for w in self.workers}),
                "rss_gb": round(sum(w.rss_gb() for w in self.workers), 1)}

    def shutdown(self) -> None:
        self._stop.set()
        for worker in self.workers:
            worker.kill()
