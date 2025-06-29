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


def find_miniseed(workdir, channel):
    for file in Path(workdir).rglob("*"):
        if file.suffix.lower() in [".msd", ".miniseed", ".mseed"]:
            try:
                st = read(str(file))
                if any(tr.stats.channel == channel for tr in st):
                    return str(file)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
    return None


def calculate_ppsd(workdir, channel, inv, tw):
    workdir = Path(workdir)
    outdir = workdir / f"npz_{channel}"
    outdir.mkdir(exist_ok=True)

    files = [
        f for f in workdir.rglob("*")
        if f.suffix.lower() in [".msd", ".miniseed", ".mseed"]
    ]

    for file in tqdm(
            files, desc=f"[{workdir.name} | {channel}] PSD files", unit="file"
            ):
        try:
            st = read(str(file))
            for trace in st.select(channel=channel):
                ppsd = PPSD(trace.stats, metadata=inv, ppsd_length=tw)
                ppsd.add(trace)
                timestamp = trace.stats.starttime.strftime(
                    '%y-%m-%d_%H-%M-%S.%f'
                    )
                outfile = outdir / f"{timestamp}.npz"
                ppsd.save_npz(str(outfile))
        except Exception as e:
            print(f"Error processing {file}: {e}")


def plot_ppsd(
        sampledata, channel, inv, npzfolder, output_folder,
        tw, units, plot_kwargs=None
        ):

    if plot_kwargs is None:
        plot_kwargs = {}

    st = read(sampledata)
    trace = st.select(channel=channel)[0]
    ppsd = PPSD(trace.stats, inv, ppsd_length=tw)

    for file in Path(npzfolder).glob("*.npz"):
        try:
            ppsd.add_npz(str(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    figfile = output_folder / f"{trace.id}.png"
    figsize = plot_kwargs.pop("figsize", (12, 6))
    dpi = plot_kwargs.pop("dpi", 300)
    cmap = plot_kwargs.pop("cmap", pqlx)
    try:
        fig = ppsd.plot(
            cmap=cmap,
            show=False,
            **plot_kwargs
        )
        fig.set_size_inches(*figsize)
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


def process_dataset(entry, tw, units):
    folder = entry["folder"]
    resp_file = entry["response"]
    channels = entry["channels"]
    output_folder = entry.get("output_folder", folder)
    action = str(entry.get("action", "full"))
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
        "figsize"
    }
    plot_kwargs = {k: entry[k] for k in PLOT_KWARGS if k in entry}

    for channel in channels:
        print(f"===> {folder} | {channel} | action={action}")
        npzfolder = Path(folder) / f"npz_{channel}"

        if action in ["calculate", "full"]:
            calculate_ppsd(folder, channel, inv, tw)

        if action in ["plot", "full"]:
            sample = find_miniseed(folder, channel)
            if sample:
                plot_ppsd(
                    sample, channel, inv, npzfolder, output_folder, tw, units,
                    plot_kwargs=plot_kwargs.copy()
                )
            else:
                print(f"No valid trace found in {folder} for {channel}")

        if action == "convert":
            convert_npz_to_text(npzfolder)


def main(config_path):
    config = load_config(config_path)
    tw = config["timewindow"]
    num_workers = config.get("num_workers", 1)
    units = config.get("units", "s").lower()
    if units == "hz":
        units = True
    else:
        units = False

    datasets = config["datasets"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_dataset, entry, tw, units)
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
