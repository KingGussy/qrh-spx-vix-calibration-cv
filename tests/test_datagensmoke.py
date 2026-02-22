import subprocess
import sys
from pathlib import Path
import numpy as np

def test_data_gen_writes_shard(tmp_path):
    out_dir = tmp_path / "synthetic_qrh_spx_vix"
    run_name = "smoke"
    cmd = [
        sys.executable, "scripts/generate_synth_qrh_spx_vix.py",
        "--out_dir", str(out_dir),
        "--run_name", run_name,
        "--n_samples", "3",
        "--shard_size", "2",
        "--m_paths", "512",
        "--n_steps_mc", "128",
        "--n_steps_vix", "32",
        "--T_list", "0.03,0.05",
        "--seed", "123",
    ]
    subprocess.check_call(cmd)

    run_dir = out_dir / run_name
    shards = sorted(run_dir.glob("*.npz"))
    assert shards, f"No shards found in {run_dir}"

    with np.load(shards[0], allow_pickle=False) as z:
        for k in ["a","b","c0","lam","eta","z0","T","spx_iv","vix_iv","spx_C","vix_C"]:
            assert k in z.files
            