from cx_Freeze import setup, Executable
import sys
import os
from pathlib import Path
import matplotlib
import obspy

mpl_data_path = matplotlib.get_data_path()
sys.setrecursionlimit(8000)

obsipy_data_dir = os.path.join(
    os.path.dirname(obspy.__file__), "imaging", "data"
    )

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
            "includes": ["matplotlib.backends.backend_agg",
                         "matplotlib.backends.backend_tkagg",
                         "matplotlib.pyplot",
                         "matplotlib.cm"],
            "packages": ["obspy", "matplotlib", "yaml", "numpy",
                         "tqdm", "os", "pathlib"],
            "excludes": [],
            "include_files": [
                (str(dist_info), f"lib/{dist_info.name}"),
                (mpl_data_path, "lib/matplotlib/mpl-data"),
                (obsipy_data_dir, "lib/obspy/imaging/data"),
                ("example", "example"),
                ("LICENSE", "LICENSE")
            ],
            "build_exe": "build/PPSD_Plotter_Windows"
        }
    },
    executables=[exe],
)
