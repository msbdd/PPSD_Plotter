import subprocess
from pathlib import Path
import shutil
import sys


def test_creates_expected_plots(tmp_path):

    config_path = Path("example/example_config.yaml")
    result_dir = Path("example/result")

    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "src/PPSD_plotter.py", str(config_path)],
        capture_output=True,
        text=True,
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0, "PPSD_Plotter failed"

    expected_files = [
        "IU.ANMO.00.BH1.png",
        "IU.ANMO.00.BH2.png",
        "IU.ANMO.00.BHZ.png",
        "IU.GRFO.00.BH1.png",
        "IU.GRFO.00.BH2.png",
        "IU.GRFO.00.BHZ.png"
    ]

    for fname in expected_files:
        file_path = result_dir / fname
        assert file_path.exists(), f"Expected file not found: {file_path}"
