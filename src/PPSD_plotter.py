import sys
from pathlib import Path
import numpy as np
import matplotlib
import yaml
from obspy import read, read_inventory, Stream
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ppsd_plotter_aux import (
    find_miniseed,
    load_inventory,
    parse_npz_timestamp,
    is_time_in_range,
    filter_npz_files_by_time,
    split_trace_by_time_filter,
    group_files_by_day,
)
matplotlib.use("Agg")

# Default PPSD time window in seconds
DEFAULT_TIME_WINDOW = 3600


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def calculate_ppsd(workdir, npzfolder, channel, location, inv, tw, time_filter=None):
    workdir = Path(workdir)
    npzfolder = Path(npzfolder)
    npzfolder.mkdir(exist_ok=True)

    files = [
        f for f in workdir.rglob("*")
        if f.suffix.lower() in [".msd", ".miniseed", ".mseed"]
    ]

    jobs = group_files_by_day(files, [(location, channel)])

    for day_files, day, loc, chan in tqdm(
            jobs, desc=f"[{workdir.name} | {channel}] PSD days", unit="day"
            ):
        st = Stream()
        for f in day_files:
            try:
                s = read(str(f))
                s = s.select(channel=chan, location=loc if loc else "")
                st += s
            except Exception as e:
                print(f"Read error in {f.name}: {e}")
        if not st:
            continue

        try:
            st.merge(method=1, interpolation_samples=1)
        except Exception as e:
            print(f"Merge error for {chan} {day}: {e}")
            continue
        st = st.split()
        if not st:
            continue

        try:
            ref_stats = st[0].stats
            ppsd_by_label = {}
            for tr in st:
                for tr_seg, label in split_trace_by_time_filter(tr, time_filter):
                    if label not in ppsd_by_label:
                        ppsd_by_label[label] = PPSD(
                            ref_stats, metadata=inv, ppsd_length=tw
                        )
                    ppsd_by_label[label].add(tr_seg)

            day_str = day.strftime("%y-%m-%d")
            for label, ppsd in ppsd_by_label.items():
                suffix = "" if label == "all" else f"_{label}"
                outfile = npzfolder / f"{day_str}{suffix}.npz"
                ppsd.save_npz(str(outfile))
        except Exception as e:
            print(
                f"Error processing {chan} location={loc or ''} {day}: {e}"
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
    
    # Check if we have labeled files (_day or _night)
    labeled_files = [f for f in all_files if f.stem.endswith('_day') or f.stem.endswith('_night')]
    unlabeled_files = [f for f in all_files if not (f.stem.endswith('_day') or f.stem.endswith('_night'))]
    
    if time_filter and unlabeled_files and not labeled_files:
        print("Warning: Using time_filter with unlabeled .npz files.")
        print("For best results, recalculate with 'action: full' or 'action: calculate' to split traces at boundaries.")
        print("Currently using timestamp-based filtering on existing files.")
    
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
            time_filter = entry.get("time_filter")
            calculate_ppsd(folder, npzfolder, channel, loc_code, inv, tw, time_filter)

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
    tw = config.get("timewindow", DEFAULT_TIME_WINDOW)
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
