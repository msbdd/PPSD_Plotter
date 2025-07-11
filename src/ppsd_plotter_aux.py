from obspy import read, read_inventory
import os
from pathlib import Path
from obspy.signal import PPSD
import sys
import numpy as np


def find_miniseed_channels(folder):
    extensions = [".mseed", ".msd", ".miniseed"]
    seen = set()
    for ext in extensions:
        for path in Path(folder).rglob(f"*{ext}"):
            try:
                st = read(str(path))
                for tr in st:
                    if len(tr) == 0:
                        continue
                    key = (tr.stats.location.strip(), tr.stats.channel.strip())
                    if key not in seen:
                        seen.add(key)
                if seen:
                    return sorted(seen)
            except Exception:
                continue
    return []


def find_miniseed(workdir, channel, location=None):
    for file in Path(workdir).rglob("*"):
        if file.suffix.lower() in [".msd", ".miniseed", ".mseed"]:
            try:
                st = read(str(file))
                for tr in st:
                    if location:
                        if (
                            tr.stats.channel == channel
                            and tr.stats.location == location
                        ):
                            return str(file)
                    else:
                        if tr.stats.channel == channel:
                            return str(file)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
    return None


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
                    npzfolder = (
                        Path(resource_path(folder)) / f"npz_{loc}_{chan}"
                    )
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


def calculate_noise_line(amplitude_mg, freq_range=(1.0, 10.0)):

    A_rms = amplitude_mg * 9.80665e-3
    f1, f2 = freq_range
    bandwidth = abs(f2 - f1)
    if bandwidth <= 0:
        raise ValueError("Frequency range must have non-zero width")

    PSD = (A_rms ** 2) / bandwidth
    dB_value = 10 * np.log10(PSD)
    return f1, f2, dB_value
