"""Tests for the /usr/bin/time + llama.cpp log-parsing in scripts/bench_efficiency.py.

Pure regex parsing, no GPU or actual llama.cpp binary needed -- but a wrong
regex here would silently corrupt the efficiency side of every Pareto-grid
point, so it's worth fixing against real-shaped sample output before it's
ever run for real.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from bench_efficiency import parse_peak_ram_gb, parse_tok_s  # noqa: E402

LLAMA_CPP_STDERR = """
llama_model_loader: loaded meta data with 30 key-value pairs
llama_print_timings:        load time =     512.34 ms
llama_print_timings:      sample time =      12.00 ms /   128 runs   (    0.09 ms per token, 10666.67 tokens per second)
llama_print_timings: prompt eval time =     890.12 ms /   512 tokens (    1.74 ms per token,   575.11 tokens per second)
llama_print_timings:        eval time =   11234.56 ms /   127 runs   (   88.46 ms per token,    11.31 tokens per second)
llama_print_timings:       total time =   12636.02 ms /   639 tokens
"""

MACOS_TIME_L = """
       12.64 real         9.87 user         2.10 sys
  2147483648  maximum resident set size
     1048576  peak memory footprint
"""

LINUX_TIME_V = """
	Command being timed: "llama-mtmd-cli -m model.gguf"
	User time (seconds): 9.87
	System time (seconds): 2.10
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.64
	Maximum resident set size (kbytes): 2097152
	Exit status: 0
"""


def test_parse_tok_s_picks_the_final_generation_line():
    # three "tokens per second" lines appear (sample/prompt-eval/eval) -- the
    # last one is the actual generation throughput, not sampling or prompt eval
    assert parse_tok_s(LLAMA_CPP_STDERR) == 11.31


def test_parse_tok_s_missing_returns_none():
    assert parse_tok_s("no timing info here") is None


def test_parse_peak_ram_macos_bytes_to_gb():
    assert parse_peak_ram_gb(MACOS_TIME_L, is_darwin=True) == 2147483648 / 1e9


def test_parse_peak_ram_linux_kbytes_to_gb():
    assert parse_peak_ram_gb(LINUX_TIME_V, is_darwin=False) == 2097152 * 1024 / 1e9


def test_parse_peak_ram_wrong_platform_format_returns_none():
    # macOS-shaped output parsed with the Linux regex must not silently match
    # something else -- absence should stay absence, not a wrong number
    assert parse_peak_ram_gb(MACOS_TIME_L, is_darwin=False) is None
    assert parse_peak_ram_gb(LINUX_TIME_V, is_darwin=True) is None
