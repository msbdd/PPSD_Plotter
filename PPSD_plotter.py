import yaml
import numpy as np
from obspy import read, read_inventory
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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

    for file in workdir.rglob("*"):
        if file.suffix.lower() in [".msd", ".miniseed", ".mseed"]:
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


def plot_ppsd(sampledata, channel, inv, npzfolder, output_folder, tw):
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
    fig = ppsd.plot(cmap=pqlx, show_mean=True, show_histogram=True,
                    xaxis_frequency=True, show_mode=True, show=False)
    fig.set_size_inches(12, 6)
    fig.savefig(figfile, dpi=300)
    print(f"Saved plot: {figfile}")


def convert_npz_to_text(npzdir):
    npzdir = Path(npzdir)
    outdir = npzdir.with_name(npzdir.name + "_text")
    outdir.mkdir(exist_ok=True)

    psd_entries = []
    periods_struct = None

    for file in npzdir.glob("*.npz"):
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
    action = str(entry.get("action", "3"))

    try:
        inv = read_inventory(resp_file)
    except Exception as e:
        print(f"Failed to read inventory {resp_file}: {e}")
        return

    for channel in channels:
        print(f"===> {folder} | {channel} | action={action}")
        npzfolder = Path(folder) / f"npz_{channel}"

        if action in ["1", "3"]:
            calculate_ppsd(folder, channel, inv, tw)

        if action in ["2", "3"]:
            sample = find_miniseed(folder, channel)
            if sample:
                plot_ppsd(sample, channel, inv, npzfolder, output_folder, tw)
            else:
                print(f"No valid trace found in {folder} for {channel}")

        if action == "4":
            convert_npz_to_text(npzfolder)


def main(config_path):
    config = load_config(config_path)
    tw = config["timewindow"]
    num_workers = config.get("num_workers", 1)

    datasets = config["datasets"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_dataset, entry, tw)
            for entry in datasets
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
