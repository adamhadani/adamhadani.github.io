---
layout: single
title: "Debugging Stuck Processes with Python 3.14's new live debugging feature and Claude Code for LLM-Assisted Introspection"
date: 2026-01-23
categories: [debugging, python, llm]
tags: [python, debugging, gevent, celery, llm, claude-code, yfinance]
excerpt: "How Python 3.14's new sys.remote_exec() combined with an LLM coding assistant helped diagnose and fix a tricky threading conflict between yfinance and gevent in a production Celery worker."
---

_This post documents a real debugging session where Python 3.14's new remote execution feature, combined with Claude Code as an interactive assistant, allowed me to diagnose and fix a production issue that would have been nearly impossible to track down with traditional debugging approaches. As always, any opinions expressed are solely my own and do not express the views or opinions of my employer. I welcome feedback and discussion on this post - find me via social links on this site!_


# Introduction

Production debugging is hard. Debugging production processes that are stuck—not crashing, not throwing errors, just... frozen — is harder. This is especially so in the case of concurrency-related issues where race conditions and deadlocks are notoriously hard to reason about. I recently had the happy ocassion to deal with such a scenario, where a combination of [cooperative multi-tasking](https://en.wikipedia.org/wiki/Cooperative_multitasking) (a la gevent), asyncio, and thread pool executors all conspired together to make for an especially potent mixture.

This post presents an approach to debugging under difficult conditions, making use of recent technologies - namely Python 3.14 and Claude Code. Here we're tracking down a subtle interaction between three technologies that individually work fine but together create a deadlock: **Celery workers using gevent pools**, **yfinance's internal threading**, and **Python's asyncio**. The bug only manifested in production workloads, not in tests, as often happens in such cases.

The hero of this story is Python 3.14's new `sys.remote_exec()` function, which lets you inject arbitrary Python code into a running process. Combined with an LLM assistant (Claude Code) that could iterate on diagnostic scripts in real-time, *what would have been days of log-and-restart debugging became a 30-minute investigation*.

_TLDR; Python 3.14's `sys.remote_exec()` enables "live" debugging of stuck processes. Pair it with an LLM that can generate targeted diagnostic scripts, and you get an interactive debugging session with a frozen process._


## The Problem

The context for this example is an algo-trading system which uses [Celery](https://docs.celeryq.dev/en/stable/) with [gevent](https://www.gevent.org/)-based pools for background tasks. One such task fetches market data from yfinance, a popular Python library for Yahoo Finance data:

```python
# Simplified task
@celery.task
def analyze_symbols(symbols: list[str]):
    df = yf.download(symbols, period="1mo")
    return analyze(df)
```

In development and unit tests this works perfectly. In production-like environment under load the worker would occasionally freeze. No error logs. No exceptions. Just... silence. The task would never complete, and eventually Celery's hard timeout (10 minutes) would SIGKILL the worker.

### Why Traditional Debugging Failed

**Log-based debugging**:  logging was added everywhere. The last log line was always right before `yf.download()`. But that told us *where* it froze, not *why*.

**Debugger attachment**: We could try `py-spy` and similar tools. They could show us stack traces, but:
- gevent greenlets don't show up as OS threads
- The process was stuck in C code (inside threading primitives), so Python-level introspection was limited

**Reproduction**: The bug was non-deterministic. It happened maybe once in 50 runs, always in the production environment with real network conditions, never in tests.

**Time pressure**: This is a mission-critical system. Every frozen worker means missing data during critical hours.


## Enter Python 3.14: `sys.remote_exec()`

Python 3.14 introduced a game-changing debugging capability: [PEP 768 – Safe external debugger interface for CPython](https://peps.python.org/pep-0768/). The star feature is `sys.remote_exec(pid, script_path)`, which injects a Python script into a running interpreter.

```python
# From another terminal:
import sys
sys.remote_exec(12345, "/path/to/debug_script.py")
```

The script executes in the target process's context, with access to all its state. Output goes to a file (since stdout isn't available), and you read the results.

**Key insight**: Unlike attaching a debugger, this doesn't require the process to be in a debuggable state. The script runs at the next "safe point" in the interpreter—and crucially, it works even when the process appears stuck.


## The Debugging Session: Human + LLM

Here's where the LLM assistance became valuable. I was using Claude Code (Anthropic's CLI tool) as my pair-debugging partner. The workflow looked like this:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Observe frozen worker                                       │
│  2. Describe symptoms to Claude Code                            │
│  3. Claude generates diagnostic script                          │
│  4. Inject script via sys.remote_exec()                         │
│  5. Read output file                                            │
│  6. Share results with Claude                                   │
│  7. Claude refines hypothesis, generates new script             │
│  8. Repeat until root cause found                               │
└─────────────────────────────────────────────────────────────────┘
```

Better yet, in cases where you can run the process locally on your development workstation and reproduce the issue (e.g. deadlock in our case), this gets even simpler - you can simply instructor claude code to find the OS process (e.g. celery worker process PID), and directly interact with it. This makes the debugging iteration loop completely human-free: Claude Code will iterate on writing scripts, injecting into process, observing the tracebacks and so forth. In my case, this pinpointed the issue in a matter of minutes.

### Round 1: Basic Stack Traces

First attempt—get stack traces for all threads:

```python
# debug_threads.py
import sys
import traceback
import threading
import io

output = io.StringIO()
output.write("\n" + "=" * 80 + "\n")
output.write("REMOTE DEBUG: Stack traces for all threads\n")
output.write("=" * 80 + "\n\n")

for thread_id, frame in sys._current_frames().items():
    thread_name = "Unknown"
    for t in threading.enumerate():
        if t.ident == thread_id:
            thread_name = t.name
            break
    output.write(f"\n--- Thread {thread_id} ({thread_name}) ---\n")
    output.write("".join(traceback.format_stack(frame)))

with open("/tmp/debug_output.txt", "w") as f:
    f.write(output.getvalue())
```

**Result**: Only ONE thread visible—the gevent hub. Of course! Gevent uses greenlets, not OS threads. `sys._current_frames()` only sees OS threads.

### Round 2: Gevent-Aware Introspection

I described the finding to Claude, and it immediately suggested looking at greenlets via the garbage collector:

```python
# debug_gevent.py
import sys
import traceback
import io
import gc

output = io.StringIO()
output.write("REMOTE DEBUG: Gevent greenlet stack traces\n\n")

try:
    import gevent
    from greenlet import greenlet as raw_greenlet

    # Get the hub info
    hub = gevent.get_hub()
    output.write(f"Hub: {hub}\n")
    output.write(f"Hub loop pendingcnt: {hub.loop.pendingcnt}\n\n")

    # Find all Greenlet objects via garbage collector
    greenlets = [obj for obj in gc.get_objects()
                 if isinstance(obj, gevent.Greenlet)]
    output.write(f"Found {len(greenlets)} Greenlet objects\n\n")

    for i, g in enumerate(greenlets):
        output.write(f"\n--- Greenlet {i}: {g!r} ---\n")
        output.write(f"  Dead: {g.dead}, Started: {g.started}\n")

        if hasattr(g, 'gr_frame') and g.gr_frame is not None:
            output.write("  Stack trace:\n")
            for line in traceback.format_stack(g.gr_frame):
                output.write("    " + line)

except Exception as e:
    output.write(f"Error: {e}\n")
    import traceback as tb
    output.write(tb.format_exc())

with open("/tmp/debug_gevent_output.txt", "w") as f:
    f.write(output.getvalue())
```

**Breakthrough!** Now I could see where each greenlet was stuck:

```
--- Greenlet 3: <Greenlet "Greenlet-0" at 0x...> ---
  Dead: False, Started: True
  Stack trace:
    File ".../celery/app/trace.py", line 451, in trace_task
    File ".../antelope/tasks/data_ingest/...", line 87, in task
    File ".../yfinance/multi.py", line 158, in download
    File ".../concurrent/futures/_base.py", line 456, in result
    File ".../threading.py", line 655, in wait
```

The greenlet was stuck in `threading.Event.wait()`—inside yfinance's internal `ThreadPoolExecutor`.

### Round 3: Understanding the Deadlock

With this clue, Claude helped me understand what was happening:

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Deadlock Pattern                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Gevent worker (greenlet)                                       │
│       │                                                         │
│       ▼                                                         │
│  yf.download()                                                  │
│       │                                                         │
│       ▼                                                         │
│  ThreadPoolExecutor.submit()  ◄─── Creates REAL OS threads      │
│       │                                                         │
│       ▼                                                         │
│  Future.result()              ◄─── Blocks on threading.Event    │
│       │                                                         │
│       ▼                                                         │
│  threading.Event.wait()       ◄─── STUCK HERE                   │
│                                                                 │
│  Problem: gevent has monkey-patched threading.Event             │
│  The ThreadPoolExecutor threads run OUTSIDE gevent's control    │
│  When they try to set() the Event, gevent's hub doesn't see it  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Root cause**: yfinance 1.0+ uses `concurrent.futures.ThreadPoolExecutor` internally for parallel downloads. When gevent monkey-patches the `threading` module, it replaces primitives like `Event` with gevent-aware versions. But `ThreadPoolExecutor` creates *real* OS threads that operate outside gevent's cooperative scheduling. When these threads signal completion via `Event.set()`, the greenlet waiting on `Event.wait()` sometimes never wakes up—a classic gevent/threading conflict.


## The Solution

Armed with the diagnosis, the fix was straightforward. Claude suggested two approaches:

### Approach 1: Disable yfinance's Internal Threading

```python
df = yf.download(
    symbols,
    threads=False,  # Disable parallel downloads
    ...
)
```

This forces sequential downloads, avoiding the ThreadPoolExecutor entirely. Slower, but reliable.

### Approach 2: Wrap with gevent.Timeout

For cases where we still want parallel behavior, wrap the call with gevent's native timeout:

```python
import gevent

TIMEOUT_SECONDS = 30

def download_with_timeout(symbols, **kwargs):
    for attempt in range(3):
        try:
            with gevent.Timeout(TIMEOUT_SECONDS):
                return yf.download(symbols, threads=False, **kwargs)
        except gevent.Timeout:
            logger.warning(f"yfinance timeout (attempt {attempt + 1})")
            gevent.sleep(1 * (2 ** attempt))  # Exponential backoff

    return pd.DataFrame()  # Empty fallback
```

We went with both: `threads=False` to prevent the deadlock, plus `gevent.Timeout` as a safety net for any other blocking behavior.


## The Meta-Lesson: LLM as Debugging Partner

What made this debugging session productive wasn't just `sys.remote_exec()`—it was the iteration speed enabled by the LLM.

Traditional debugging workflow:
1. Hypothesize
2. Write diagnostic code
3. Deploy
4. Wait for reproduction
5. Analyze results
6. Repeat

Each cycle: 15-60 minutes minimum.

LLM-assisted workflow:
1. Describe observation
2. LLM generates diagnostic script immediately
3. Inject script (10 seconds)
4. Read results
5. Share with LLM
6. LLM refines hypothesis
7. Repeat

Each cycle: 2-5 minutes.

The LLM's value wasn't that it "knew the answer"—it didn't. The value was:

- **Instant script generation**: No time spent writing boilerplate for introspection
- **Domain knowledge synthesis**: Knew to check `gc.get_objects()` for gevent greenlets
- **Hypothesis refinement**: Each result led to a more targeted next query
- **Pattern recognition**: Quickly identified the gevent/threading conflict pattern

This is a new debugging modality: **interactive introspection with an AI pair**. The LLM doesn't need to understand your entire codebase—it just needs to help you interrogate the live process state efficiently.


## Practical Guide: Remote Debugging Setup

For those wanting to replicate this workflow:

### Prerequisites

- Python 3.14+ on both debugger and target
- `sudo` access on macOS (for `com.apple.system-task-ports` entitlement). This is a hard requirement - the OS user with which you run Claude Code needs to be able to use `sudo` seemlessly. This can be achieved for example by adding the user to /etc/sudoers via `visudo`, see for example [here](https://apple.stackexchange.com/a/388962/495866).
- Target process must be Python (obviously)

### Quick Start

```bash
# 1. Find your stuck process
pgrep -f "celery.*worker"

# 2. Create debug script
cat > /tmp/debug.py << 'EOF'
import sys, traceback, io
output = io.StringIO()
for tid, frame in sys._current_frames().items():
    output.write(f"\n--- Thread {tid} ---\n")
    output.write("".join(traceback.format_stack(frame)))
with open("/tmp/debug_out.txt", "w") as f:
    f.write(output.getvalue())
EOF

# 3. Inject (replace PID)
sudo python3.14 -c "import sys; sys.remote_exec(12345, '/tmp/debug.py')"

# 4. Read results
cat /tmp/debug_out.txt
```

### Gevent-Specific Script

For gevent workers, use the enhanced script from Round 2 above.

### Caveats

- Script execution is **asynchronous**—it runs at the next safe point
- If the process is blocked in C code, the script won't run until Python resumes
- Output must go to files; stdout/stderr aren't captured
- The script runs with the target's import context—use absolute imports


## Conclusion

Python 3.14's `sys.remote_exec()` is a game-changer for production debugging. It transforms stuck processes from black boxes into queryable systems. Combined with an LLM that can rapidly generate targeted diagnostic scripts, you get an interactive debugging session with a frozen process—something that was simply not possible before.

The specific bug we found—gevent/yfinance threading conflict—is well-known in retrospect. But finding it required:

1. Knowing to look at greenlets, not threads
2. Knowing to use `gc.get_objects()` to find them
3. Understanding the gevent monkey-patching implications
4. Recognizing the ThreadPoolExecutor pattern

An LLM assistant, even without knowing our specific codebase, could help navigate all of this because it has broad knowledge of these libraries and their interaction patterns. The human brings the context (what the system is supposed to do, what's broken), and the LLM brings the diagnostic techniques and pattern recognition.

This is what AI-assisted development looks like in practice: not replacing human judgment, but dramatically accelerating the iteration loop on complex diagnostic tasks.

---

## Appendix: Full Diagnostic Script

Here's the complete gevent-aware diagnostic script we developed:

```python
"""
Remote debugging script for gevent-based Celery workers.
Inject via: sudo python3.14 -c "import sys; sys.remote_exec(PID, 'this_script.py')"
Output written to: /tmp/debug_gevent_output.txt
"""
import sys
import traceback
import io
import gc
from datetime import datetime

output = io.StringIO()
output.write(f"Debug snapshot at {datetime.now().isoformat()}\n")
output.write("=" * 80 + "\n\n")

# Section 1: OS Threads
output.write("=== OS THREADS (sys._current_frames) ===\n")
import threading
for thread_id, frame in sys._current_frames().items():
    thread_name = "Unknown"
    for t in threading.enumerate():
        if t.ident == thread_id:
            thread_name = t.name
            break
    output.write(f"\n--- Thread {thread_id} ({thread_name}) ---\n")
    output.write("".join(traceback.format_stack(frame)))

# Section 2: Gevent Greenlets
output.write("\n\n=== GEVENT GREENLETS ===\n")
try:
    import gevent
    from greenlet import greenlet as raw_greenlet

    hub = gevent.get_hub()
    output.write(f"Hub: {hub}\n")
    # Use pendingcnt (not pending) - correct attribute for libev loop
    output.write(f"Hub loop pendingcnt: {hub.loop.pendingcnt}\n")
    output.write(f"Hub loop activecnt: {hub.loop.activecnt}\n\n")

    # Find Greenlet objects via GC
    greenlets = [obj for obj in gc.get_objects()
                 if isinstance(obj, gevent.Greenlet)]
    output.write(f"Active Greenlets: {len(greenlets)}\n\n")

    for i, g in enumerate(greenlets):
        output.write(f"--- Greenlet {i}: {g!r} ---\n")
        output.write(f"    dead={g.dead}, started={g.started}, ready={g.ready()}\n")
        if hasattr(g, 'gr_frame') and g.gr_frame is not None:
            output.write("    Stack:\n")
            for line in traceback.format_stack(g.gr_frame):
                output.write("      " + line)
        output.write("\n")

    # Raw greenlets (hub, etc.)
    raw_greenlets = [obj for obj in gc.get_objects()
                     if isinstance(obj, raw_greenlet)
                     and not isinstance(obj, gevent.Greenlet)]
    output.write(f"\nRaw greenlets (hub/internal): {len(raw_greenlets)}\n")

except ImportError:
    output.write("gevent not available\n")
except Exception as e:
    output.write(f"Error inspecting gevent: {e}\n")
    output.write(traceback.format_exc())

# Section 3: Celery Task State
output.write("\n\n=== CELERY STATE ===\n")
try:
    from celery import current_task
    if current_task:
        output.write(f"Current task: {current_task.name}\n")
        output.write(f"Task ID: {current_task.request.id}\n")
    else:
        output.write("No current task\n")
except Exception as e:
    output.write(f"Celery inspection failed: {e}\n")

# Write output
with open("/tmp/debug_gevent_output.txt", "w") as f:
    f.write(output.getvalue())

print("Debug output written to /tmp/debug_gevent_output.txt")
```

### Example Output

Here's what the output looks like when run against an idle Celery worker with gevent pool:

```
Debug snapshot at 2026-01-23T11:24:28.895997
================================================================================

=== OS THREADS (sys._current_frames) ===

--- Thread 8533978176 (Unknown) ---
  File ".../gevent/hub.py", line 647, in run
    loop.run()
  File "<string>", line 26, in <module>


=== GEVENT GREENLETS ===
Hub: <Hub '' at 0x10f546610 select default pending=0 ref=7
     resolver=<gevent.resolver.thread.Resolver at 0x119d97770
     pool=<ThreadPool at 0x119ccf530 tasks=0 size=0 maxsize=10>>>
Hub loop pendingcnt: 0
Hub loop activecnt: 7

Active Greenlets: 4

--- Greenlet 0: <_Greenlet at 0x11a45a660: _run> ---
    dead=True, started=False, ready=True

--- Greenlet 1: <_Greenlet at 0x11a45a520: _run> ---
    dead=True, started=False, ready=True

--- Greenlet 2: <_Greenlet at 0x11a45aa20: <TimerEntry: periodic(*[], **{})> ---
    dead=False, started=True, ready=False

--- Greenlet 3: <_Greenlet at 0x11a45a700: <TimerEntry: _send(*('worker-heartbeat',), **{})> ---
    dead=False, started=True, ready=False


Raw greenlets (hub/internal): 3


=== CELERY STATE ===
No current task
```

When a worker is **stuck**, you'd see greenlets with `dead=False, started=True` and stack traces pointing to the blocking code—for example, `threading.Event.wait()` inside yfinance's `ThreadPoolExecutor`.

---

**Repository**: The trading system discussed is private, but the debugging patterns discussed above are applicable to any Python-based system where you can use Python 3.14. 

**Further Reading**:
- [PEP 768 – Safe external debugger interface for CPython](https://peps.python.org/pep-0768/)
- [gevent documentation on monkey patching](https://www.gevent.org/intro.html#monkey-patching)
- [yfinance threading discussion](https://github.com/ranaroussi/yfinance/issues)

