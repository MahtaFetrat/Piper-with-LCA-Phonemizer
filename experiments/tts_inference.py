# benchmark.py
import os
import time
import subprocess
from datetime import datetime

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
log_file   = "piper_log.txt"      # all individual outputs go here

# (optional) start with a clean log for this run
# open(log_file, "w", encoding="utf-8").close()

# For the demo we just use a tiny list:
graphemes = []
with open('data.txt', 'r', encoding="utf-8") as f:
    graphemes = f.readlines()

# ------------------------------------------------------------------
# INFERENCE LOOP
# ------------------------------------------------------------------
start_time = time.time()

for idx, text in enumerate(graphemes, start=1):
    # Build the output WAV path (one file per utterance)

    # ------------------------------------------------------------------
    # Build the Piper command
    # ------------------------------------------------------------------
    #   py -m piper  -m fa_IR-amir-medium.onnx  -f <wav>  --debug -- "<text>"
    # ------------------------------------------------------------------
    cmd = [
        "py",                     # or "python" / "python3" depending on your env
        "-X", "utf8",
        "-m", "piper",
        "-m", "fa_IR-amir-medium.onnx",
        "-f", "test.wav",
        "--debug",
        "--",                     # separates Piper options from the text
        text
    ]

    print(f"[{idx}/{len(graphemes)}] Running: {' '.join(cmd)}")

    # Run Piper and capture **both** stdout & stderr into the log file
    with open(log_file, "a", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=log,
            text=True,            # work with strings, not bytes
            check=False           # don't raise on non-zero exit (Piper may warn)
        )

    # Optional: raise if Piper actually failed (exit code != 0)
    if result.returncode != 0:
        print(f"Warning: Piper returned {result.returncode} for utterance {idx}")

# ------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------
total_time = time.time() - start_time
avg_time   = total_time / len(graphemes) if graphemes else 0

summary = f"""
=== Piper Benchmark Summary ===
Start time : {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
End time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total texts: {len(graphemes)}
Total time : {total_time:.2f} s
Avg time   : {avg_time:.3f} s per utterance
Log file   : {os.path.abspath(log_file)}
"""

print(summary)

# Also append the summary to the log
with open(log_file, "a", encoding="utf-8") as log:
    log.write(summary)