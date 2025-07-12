from cx_Freeze import setup, Executable
import sys
import os
from pathlib import Path
import matplotlib
import obspy

mpl_data_path = matplotlib.get_data_path()
sys.setrecursionlimit(8000)
sys.path.insert(0, 'src')

obsipy_data_dir = os.path.join(
    os.path.dirname(obspy.__file__), "imaging", "data"
    )

site_packages = next(p for p in sys.path if 'site-packages' in p)
dist_info = next(Path(site_packages).glob("obspy-*.dist-info"))

exe = Executable(
    script=os.path.join("src", "gui.py"),
    base="Win32GUI" if sys.platform == "win32" else None,
    target_name="PPSD_Plot_GUI",
    icon="resources/icon.ico"
)

setup(
    name="PPSD_Plotter",
    version="0.2.0",
    description="A simple PPSD plotting tool based on ObsPy",
    options={
        "build_exe": {
            "packages": ["obspy", "matplotlib", "yaml", "numpy",
                         "tqdm", "os", "pathlib"],
            "excludes": [],
            "includes": ["localization_dicts",
                         "ppsd_plotter_aux"],
            "include_files": [
                (str(dist_info), f"lib/{dist_info.name}"),
                (mpl_data_path, "lib/matplotlib/mpl-data"),
                (obsipy_data_dir, "lib/obspy/imaging/data"),
                ("example", "example"),
                ("LICENSE", "LICENSE"),
                ("resources/icon.ico", "resources/icon.ico"),
                ("resources/icon.png", "resources/icon.png"),
            ],
            "build_exe": "build/PPSD_Plotter_Windows_GUI"
        }
    },
    executables=[exe],
)
