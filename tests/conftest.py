import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA/GPU and built extension")