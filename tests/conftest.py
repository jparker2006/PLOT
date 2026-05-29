"""Pytest session setup.

The test suite imports BOTH the LightGBM baseline (its own OpenMP runtime, the brew ``libomp`` on
macOS) and the torch sequence model (torch's bundled OpenMP) in a single process. Two OpenMP
runtimes co-resident can abort/segfault on macOS; this flag tells OpenMP to tolerate the duplicate.
Set before either library is imported (conftest is imported first by pytest). Production runs keep
the two backends in separate processes, so they never need this.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
