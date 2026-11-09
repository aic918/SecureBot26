"""Run independent planning candidates on several CPU cores (no ROS).

The planners try many candidates that do not depend on each other (17 lane angles, 13 grid
angles for squaring) and keep the best; each is computed exactly as it would be one after
another, so the plans are the same, only sooner (2026-11-07: a hallway plan took ~70 s on
one core of the Jetson's six).

Workers are fresh processes ('spawn'), not forks of the caller: the laps node is a
multi-threaded ROS process, and forking one can hang. They stay up between plans.
Off by default (SECUREBOT_PLAN_WORKERS=4 turns it on; not yet tested on the robot): off,
or if the workers cannot be started, everything runs here, one after another.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

_POOL = None
_BROKEN = False


def workers():
    try:
        n = int(os.environ.get('SECUREBOT_PLAN_WORKERS', '0'))
    except ValueError:
        n = 4
    return max(0, min(n, (os.cpu_count() or 1) - 1))


def pmap(fn, jobs):
    """[fn(job) for job in jobs], in order, on the worker processes when there are several
    jobs and workers; here otherwise (or if the workers fail)."""
    global _POOL, _BROKEN
    jobs = list(jobs)
    if len(jobs) < 2 or _BROKEN or workers() < 2:
        return [fn(job) for job in jobs]
    try:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(max_workers=workers(),
                                        mp_context=multiprocessing.get_context('spawn'))
        return list(_POOL.map(fn, jobs))
    except Exception:                                   # noqa: BLE001 (any pool trouble)
        _BROKEN = True
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None
        return [fn(job) for job in jobs]
