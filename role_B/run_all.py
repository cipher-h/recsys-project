import subprocess, sys

scripts = [
    "models/popularity.py",
    "models/svd_model.py",
    "models/als_model.py",
]

for s in scripts:
    print(f"\n{'='*40}\nRunning {s}\n{'='*40}")
    result = subprocess.run([sys.executable, s])
    if result.returncode != 0:
        print(f"ERROR in {s}, stopping.")
        break