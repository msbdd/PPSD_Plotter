from obspy import read, read_inventory
import os
from pathlib import Path
from obspy.signal import PPSD
import sys
import numpy as np
from datetime import datetime


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


def parse_npz_timestamp(filename):
    """
    Parse timestamp from npz filename.
    Format: yy-mm-dd_HH-MM-SS.ffffff.npz
    Returns datetime object or None if parsing fails.
    """
    try:
        stem = Path(filename).stem  # Remove .npz extension
        # Parse the timestamp: yy-mm-dd_HH-MM-SS.ffffff
        dt = datetime.strptime(stem, '%y-%m-%d_%H-%M-%S.%f')
        return dt
    except Exception:
        return None


def is_time_in_range(dt, start_time, end_time):
    """
    Check if datetime's time component falls within start_time and end_time.
    Handles ranges that span midnight (e.g., 22:00 to 06:00).

    Args:
        dt: datetime object
        start_time: time object (e.g., time(22, 0))
        end_time: time object (e.g., time(6, 0))

    Returns:
        True if dt.time() is within the range, False otherwise
    """
    if start_time is None or end_time is None:
        return True

    t = dt.time()

    # Normal range (e.g., 06:00 to 22:00)
    if start_time <= end_time:
        return start_time <= t <= end_time
    # Range spans midnight (e.g., 22:00 to 06:00)
    else:
        return t >= start_time or t <= end_time


def filter_npz_files_by_time(npz_files, time_filter):
    """
    Filter npz files based on time_filter configuration.

    Args:
        npz_files: list of Path objects
        time_filter: dict with optional 'night_start' and 'night_stop' keys
                    (e.g., {'night_start': '22:00', 'night_stop': '06:00'})

    Returns:
        filtered list of Path objects
    """
    if not time_filter:
        return npz_files

    night_start_str = time_filter.get('night_start')
    night_stop_str = time_filter.get('night_stop')

    if not night_start_str or not night_stop_str:
        return npz_files

    try:
        # Parse time strings (e.g., "22:00" or "22:00:00")
        night_start = datetime.strptime(night_start_str, '%H:%M').time()
        night_stop = datetime.strptime(night_stop_str, '%H:%M').time()
    except ValueError:
        try:
            # Try with seconds
            night_start = datetime.strptime(night_start_str, '%H:%M:%S').time()
            night_stop = datetime.strptime(night_stop_str, '%H:%M:%S').time()
        except ValueError:
            print("Warning: Invalid time format in time_filter. "
                  "Using all files.")
            return npz_files

    filtered = []
    for file in npz_files:
        dt = parse_npz_timestamp(file.name)
        if dt and is_time_in_range(dt, night_start, night_stop):
            filtered.append(file)

    return filtered


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
