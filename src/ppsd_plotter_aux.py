from obspy import read, read_inventory
import os
from pathlib import Path
from obspy.signal import PPSD
import sys


def calculate_ppsd_worker(job_list, inv_path, tw, folder):
    inv = load_inventory(inv_path)

    for file, loc, chan in job_list:
        try:
            st = read(str(file))
            st = st.select(channel=chan, location=loc if loc else "")
            if not st:
                continue
        except Exception as e:
            print(f"Read error in {file.name}: {e}")
            continue

        for tr in st:
            try:
                if loc:
                    npzfolder = Path(resource_path(folder)) / f"npz_{loc}_{chan}"
                else:
                    npzfolder = Path(resource_path(folder)) / f"npz_{chan}"

                npzfolder.mkdir(exist_ok=True)
                ppsd = PPSD(tr.stats, metadata=inv, ppsd_length=tw)
                ppsd.add(tr)
                timestamp = tr.stats.starttime.strftime("%y-%m-%d_%H-%M-%S.%f")
                outfile = npzfolder / f"{timestamp}.npz"
                ppsd.save_npz(str(outfile))
            except Exception as e:
                print(
                    f"[{os.getpid()}] Error processing {file.name}"
                    f" trace {tr.id}: {e}"
                    )


def load_inventory(resp_file):
    ext = Path(resp_file).suffix.lower()

    if ext in [".seed", ".dataless"]:
        fmt = "SEED"
    elif ext == ".xml":
        fmt = "STATIONXML"
    else:
        fmt = None

    try:
        if fmt:
            inv = read_inventory(resp_file, format=fmt)
        else:
            inv = read_inventory(resp_file)
        return inv
    except Exception as e:
        print(f"Failed to read inventory {resp_file}: {e}")
        return


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))
