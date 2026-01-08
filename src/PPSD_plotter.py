import sys
from pathlib import Path
import numpy as np
import matplotlib
import yaml
from obspy import read, read_inventory
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime
matplotlib.use("Agg")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_inventory(resp_file):
    ext = Path(resp_file).suffix.lower()

    if ext in ['.seed', '.dataless']:
        fmt = "SEED"
    elif ext == '.xml':
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


def parse_channel(ch_str):
    parts = ch_str.split(".", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "true", "1", "on")
    return False


def find_miniseed(workdir, channel, location=None):
    for file in Path(workdir).rglob("*"):
        if file.suffix.lower() in [".msd", ".miniseed", ".mseed"]:
            try:
                st = read(str(file))
                for tr in st:
                    if location:
                        if (
                            tr.stats.channel == channel and
                            tr.stats.location == location
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


def calculate_ppsd(workdir, npzfolder, channel, location, inv, tw):
    workdir = Path(workdir)
    Path(npzfolder).mkdir(exist_ok=True)

    files = [
        f for f in workdir.rglob("*")
        if f.suffix.lower() in [".msd", ".miniseed", ".mseed"]
    ]

    for file in tqdm(
            files, desc=f"[{workdir.name} | {channel}] PSD files", unit="file"
            ):
        try:
            st = read(str(file))
            st = st.select(channel=channel, location=location)
            for trace in st:
                ppsd = PPSD(trace.stats, metadata=inv, ppsd_length=tw)
                ppsd.add(trace)
                timestamp = trace.stats.starttime.strftime(
                    '%y-%m-%d_%H-%M-%S.%f'
                    )
                outfile = npzfolder / f"{timestamp}.npz"
                ppsd.save_npz(str(outfile))
        except Exception as e:
            print(
                f"Error processing {file} for channel={channel}"
                f"location={location}: {e}"
                )


def plot_ppsd(
        sampledata, channel, location, inv, npzfolder, output_folder,
        tw, plot_kwargs=None, time_filter=None
        ):

    if plot_kwargs is None:
        plot_kwargs = {}

    st = read(sampledata)
    if location:
        matches = st.select(channel=channel, location=location)
    else:
        matches = st.select(channel=channel)
    if not matches:
        print(f"No matching trace for channel={channel} location={location}")
        return
    if location is None and len(matches) > 1:
        print(
            f"Warning: Multiple locations found for {channel}."
            f"Using first: {matches[0].stats.location}"
            )
    trace = matches[0]
    ppsd = PPSD(trace.stats, inv, ppsd_length=tw)

    # Get all npz files and filter by time if needed
    all_files = list(Path(npzfolder).glob("*.npz"))
    filtered_files = filter_npz_files_by_time(all_files, time_filter)

    if time_filter:
        nfiltered = len(filtered_files)
        ntotal = len(all_files)
        print(f"Time filter applied: {nfiltered}/{ntotal} files selected")

    for file in filtered_files:
        try:
            ppsd.add_npz(str(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    figfile = output_folder / f"{trace.id}.png"

    cmap = plot_kwargs.pop("cmap", pqlx)
    figsize = plot_kwargs.pop("figsize", (12, 6))
    dpi = plot_kwargs.pop("dpi", 300)

    try:
        fig = ppsd.plot(
            cmap=cmap,
            show=False,
            **plot_kwargs
        )
        fig.set_size_inches(figsize)
        fig.savefig(figfile, dpi=dpi)
        print(f"Saved plot: {figfile}")
    except Exception as e:
        print(f"Error: {e}")


def convert_npz_to_text(npzdir):
    npzdir = Path(npzdir)
    outdir = npzdir.with_name(npzdir.name + "_text")
    outdir.mkdir(exist_ok=True)

    psd_entries = []
    periods_struct = None
    files = list(npzdir.glob("*.npz"))

    for file in tqdm(files, desc=f"[{npzdir.name}] Converting", unit="file"):
        data = np.load(file, allow_pickle=True)
        periods = np.asarray(data["_period_binning"]).flatten()
        psd_values = np.asarray(data["_binned_psds"]).astype(float)

        if periods_struct is None:
            periods_struct = periods

        for i, psd_row in enumerate(psd_values):
            psd_entries.append((i, psd_row.flatten()))

    if psd_entries:
        outcsv = outdir / "export.csv"
        with open(outcsv, "w") as fo:
            header = (
                "Period binning," +
                ",".join(f"{float(p):.6f}s" for p in periods_struct)
            )
            fo.write(header + "\n")
            for time_window, row in psd_entries:
                fo.write(
                    f"{time_window}," +
                    ",".join(f"{float(v):.6f}" for v in row) + "\n"
                )
        print(f"Saved CSV to {outcsv}")
    else:
        print("No PSD entries found.")


def process_dataset(entry, tw):
    folder = entry["folder"]
    resp_file = entry["response"]
    channels = entry["channels"]
    output_folder = entry.get("output_folder", folder)
    action = str(entry.get("action", "full"))
    # Use timewindow from entry if available, otherwise use global tw
    tw = entry.get("timewindow", tw)
    inv = load_inventory(resp_file)

    if not inv:
        print(f"Failed to read inventory {resp_file}")
        return

    PLOT_KWARGS = {
        "show_coverage",
        "show_percentiles",
        "show_histogram",
        "percentiles",
        "show_noise_models",
        "show_earthquakes",
        "grid",
        "max_percentage",
        "period_lim",
        "show_mode",
        "show_mean",
        "cmap",
        "cumulative",
        "cumulative_number_of_colors",
        "xaxis_frequency",
        "dpi",
        "figsize",
    }

    BOOLEAN_KEYS = {
        "show_coverage", "show_percentiles", "show_histogram",
        "show_noise_models", "show_earthquakes", "grid",
        "show_mode", "show_mean", "cumulative", "xaxis_frequency"
    }

    plot_kwargs = {}

    for k in PLOT_KWARGS:
        if k in entry:
            val = entry[k]
            if k in BOOLEAN_KEYS:
                plot_kwargs[k] = safe_bool(val)
            else:
                plot_kwargs[k] = val

    for ch_str in channels:
        loc_code, channel = parse_channel(ch_str)
        print(f"===> {folder} | {loc_code}.{channel} | action={action}")
        if loc_code:
            npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
        else:
            npzfolder = Path(folder) / f"npz_{channel}"

        if action in ["calculate", "full"]:
            calculate_ppsd(folder, npzfolder, channel, loc_code, inv, tw)

        if action in ["plot", "full"]:
            sample = find_miniseed(folder, channel, loc_code)
            if sample:
                time_filter = entry.get("time_filter")
                plot_ppsd(
                    sample, channel, loc_code, inv, npzfolder,
                    output_folder, tw, plot_kwargs=plot_kwargs.copy(),
                    time_filter=time_filter
                )
            else:
                print(f"No valid trace found in {folder} for {channel}")

        if action == "convert":
            convert_npz_to_text(npzfolder)


def main(config_path):
    config = load_config(config_path)
    tw = config.get("timewindow", 3600)  # Default to 3600 if not specified
    num_workers = config.get("num_workers", 1)
    datasets = config["datasets"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_dataset, entry, tw)
            for entry in tqdm(
                datasets, desc="Submitting tasks", unit="dataset"
                )
        ]
        for future in tqdm(futures, desc="Processing datasets", unit="task"):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")
                raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
