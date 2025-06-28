from cx_Freeze import setup, Executable
import sys
import os
from pathlib import Path

sys.setrecursionlimit(8000)

site_packages = next(p for p in sys.path if 'site-packages' in p)
dist_info = next(Path(site_packages).glob("obspy-*.dist-info"))

exe = Executable(
    script=os.path.join("src", "PPSD_plotter.py"),
    base=None if sys.platform == "win32" else None,
    target_name="PPSD_Plot"
)

setup(
    name="PPSD_Plotter",
    version="__VERSION__",
    description="A simple PPSD plotting tool based on ObsPy",
    options={
        "build_exe": {
            "packages": ["obspy"],
            "excludes": [],
            "include_files": [
                (str(dist_info), f"lib/{dist_info.name}"),
                ("example", "example"),
                ("LICENSE", "LICENSE")
            ],
            "build_exe": "build/PPSD_Plotter_Windows"
        }
    },
    executables=[exe],
)
