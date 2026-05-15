import subprocess
from pathlib import Path
import shutil
import sys


def test_creates_expected_plots():
    config_path = Path("example/example_config.yaml")
    anmo_dir = Path("example/IU.ANMO..D")
    grfo_dir = Path("example/IU.GRFO..D")

    # Clean artifacts from any prior run so we test current behavior, not
    # stale NPZs (which can mismatch the current ppsd_length and fail to load).
    for d in (anmo_dir, grfo_dir):
        for npz_dir in d.glob("npz_*"):
            shutil.rmtree(npz_dir)
        for png in d.glob("*.png"):
            png.unlink()

    result = subprocess.run(
        [sys.executable, "src/PPSD_plotter.py", str(config_path)],
        capture_output=True,
        text=True,
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0, "PPSD_Plotter failed"

    # The bundled ANMO miniseed only contains BHZ; GRFO contains all three.
    expected_files = [
        anmo_dir / "IU.ANMO.00.BHZ.png",
        grfo_dir / "IU.GRFO.00.BH1.png",
        grfo_dir / "IU.GRFO.00.BH2.png",
        grfo_dir / "IU.GRFO.00.BHZ.png",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Expected file not found: {file_path}"
