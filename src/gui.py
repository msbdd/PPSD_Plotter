import tkinter as tk
from tqdm import tqdm
import numpy as np
from tkinter import filedialog
from tkinter import ttk
import yaml
from pathlib import Path
from matplotlib import colormaps
from obspy import read_inventory
from obspy import read
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
import matplotlib
from functools import partial
import os
from tkinter import messagebox
import copy

matplotlib.use("TkAgg")

CMAP_NAMES = ["pqlx"] + sorted(colormaps)

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
}

BOOLEAN_KEYS = {
    "show_coverage",
    "show_percentiles",
    "show_histogram",
    "show_noise_models",
    "grid",
    "show_mode",
    "show_mean",
    "cumulative",
    "xaxis_frequency",
}

PARAM_LABELS = {
    "folder": "Data Folder",
    "response": "Response File",
    "channels": "Channels",
    "show_coverage": "Show Coverage",
    "show_percentiles": "Show Percentiles",
    "show_histogram": "Show Histogram",
    "percentiles": "Percentiles",
    "show_noise_models": "Show Noise Models",
    "show_earthquakes": "Show Earthquakes",
    "grid": "Show Grid",
    "max_percentage": "Max Percentage",
    "period_lim": "Period Limits",
    "show_mode": "Show Mode",
    "show_mean": "Show Mean",
    "cmap": "Colormap",
    "cumulative": "Cumulative",
    "cumulative_number_of_colors": "Cumulative Colors",
    "xaxis_frequency": "X Axis Frequency",
    "action": "Action",
    "timewindow": "Timewindow",
}

DEFAULT_PLOT_KWARGS = {
    "show_coverage": True,
    "show_histogram": True,
    "show_percentiles": False,
    "show_noise_models": True,
    "grid": True,
    "show_mode": False,
    "show_mean": False,
    "cmap": "pqlx",
    "cumulative": False,
    "xaxis_frequency": False,
}

DEFAULT_DATASET = {
    "folder": "",
    "response": "",
    "channels": [],
    "action": "full",
    "timewindow": 600,
    "plot_kwargs": DEFAULT_PLOT_KWARGS.copy(),
}


PARAM_TOOLTIPS = {
    "channels": "List of channels (e.g. BHZ or 00.BHZ, one per line).",
    "action": "What to do: plot, calculate, full, or convert.",
    "timewindow": "Window length in seconds for PPSD calculation.",
    "show_coverage": "Show data coverage on the plot.",
    "show_percentiles": "Show percentiles on the plot.",
    "show_histogram": "Show histogram on the plot.",
    "percentiles": "Percentile values to plot (comma-separated).",
    "show_noise_models": "Show noise models (NLNM/NHNM).",
    "show_earthquakes": "Show earthquakes on the plot.",
    "grid": "Show grid lines.",
    "max_percentage": "Maximum percentage for color scale.",
    "period_lim": "Limits for period axis (e.g. 0.01, 179).",
    "show_mode": "Show mode value.",
    "show_mean": "Show mean value.",
    "cmap": "Colormap for the plot.",
    "cumulative": "Show cumulative distribution.",
    "cumulative_number_of_colors": "Number of colors for cumulative plot.",
    "xaxis_frequency": "Show frequency instead of period on x-axis.",
}

ACTIONS = ["plot", "calculate", "full", "convert"]


def parse_channel(ch_str):
    parts = ch_str.split(".", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def find_miniseed_channels(folder):
    extensions = [".mseed", ".msd", ".miniseed"]
    seen = set()
    for ext in extensions:
        for path in Path(folder).rglob(f"*{ext}"):
            try:
                st = read(str(path), headonly=True)
                for tr in st:
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


def calculate_ppsd(workdir, npzfolder, channel, location, inv, tw):
    workdir = Path(workdir)
    Path(npzfolder).mkdir(exist_ok=True)

    files = [
        f
        for f in workdir.rglob("*")
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
                    "%y-%m-%d_%H-%M-%S.%f")
                outfile = npzfolder / f"{timestamp}.npz"
                ppsd.save_npz(str(outfile))
        except Exception as e:
            print(
                f"Error processing {file} for channel={channel}"
                f"location={location}: {e}"
            )


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
            header = "Period binning," + ",".join(
                f"{float(p):.6f}s" for p in periods_struct
            )
            fo.write(header + "\n")
            for time_window, row in psd_entries:
                fo.write(
                    f"{time_window}," +
                    ",".join(
                        f"{float(v):.6f}" for v in row) +
                    "\n")
        print(f"Saved CSV to {outcsv}")
    else:
        print("No PSD entries found.")


def make_yaml_safe(obj):
    if isinstance(obj, dict):
        return {k: make_yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_yaml_safe(i) for i in obj]
    else:
        return obj


def normalize_plot_kwargs(plot_kwargs):
    normalized = {}
    for key, val in plot_kwargs.items():
        if key in BOOLEAN_KEYS:
            normalized[key] = safe_bool(val)
        elif key == "cmap" and isinstance(val, str):
            normalized[key] = val
        else:
            normalized[key] = val
    return normalized


def process_dataset_visual(ds, tw, progress_update_callback):
    folder = ds.get("folder", "")
    resp_file = ds.get("response", "")
    channels = ds.get("channels", [])
    action = str(ds.get("action", "full"))
    inv = load_inventory(resp_file)

    if not inv:
        progress_update_callback(0, "Failed: inventory load")
        return

    plot_kwargs = normalize_plot_kwargs(ds.get("plot_kwargs", {}))
    total = len(channels)
    if total == 0:
        progress_update_callback(0, "No channels defined")
        return

    for i, ch_str in enumerate(channels):
        progress = int((i / total) * 100)
        progress_update_callback(progress, f"Processing {ch_str}")

        loc_code, channel = parse_channel(ch_str)

        if loc_code:
            npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
        else:
            npzfolder = Path(folder) / f"npz_{channel}"

        if action in ["calculate", "full"]:
            calculate_ppsd(folder, npzfolder, channel, loc_code, inv, tw)

        if action in ["plot", "full"]:
            sample = find_miniseed(folder, channel, loc_code)
            if sample:
                plot_ppsd_interactive(
                    sample,
                    channel,
                    loc_code,
                    inv,
                    npzfolder,
                    tw,
                    plot_kwargs.copy())
            else:
                progress_update_callback(progress, f"No data for {ch_str}")

        if action == "convert":
            convert_npz_to_text(npzfolder)

    progress_update_callback(100, "Done")


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


def format_plot_kwargs_for_display(plot_kwargs):
    lines = []
    for key, value in plot_kwargs.items():
        label = PARAM_LABELS.get(key, key)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def plot_ppsd_interactive(
    sampledata, channel, location, inv, npzfolder, tw, plot_kwargs=None
):
    if plot_kwargs is None:
        plot_kwargs = {}
    cmap_name_or_obj = plot_kwargs.pop("cmap", "pqlx")
    if isinstance(cmap_name_or_obj, str):
        if cmap_name_or_obj == "pqlx":
            cmap = pqlx
        else:
            cmap = colormaps.get(cmap_name_or_obj, "viridis")  # fallback
    else:
        cmap = cmap_name_or_obj  # in case already a colormap object
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

    for file in Path(npzfolder).glob("*.npz"):
        try:
            ppsd.add_npz(str(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    fig = ppsd.plot(cmap=cmap, show=False, **plot_kwargs)

    fig.canvas.manager.set_window_title(f"PPSD Plot {trace.id}")
    fig.show()


def safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, _, cy = (
            self.widget.bbox("insert") if hasattr(
                self.widget, "bbox") else (
                0, 0, 0, 0))
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class DatasetFrame(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        dataset,
        index,
        run_callback=None,
        delete_callback=None,
        duplicate_callback=None,
    ):
        super().__init__(parent, text=f"Dataset {index+1}", padding=5)
        self.dataset = dataset
        self.index = index
        self.run_callback = run_callback
        self.delete_callback = delete_callback
        self.duplicate_callback = duplicate_callback
        self.plot_kwargs_vars = {}
        self.dataset.setdefault("plot_kwargs", {})
        self.build()

    def build_path_selector(self, label_text, key, row):
        def select_path():
            path = (
                filedialog.askdirectory()
                if "folder" in key
                else filedialog.askopenfilename()
            )
            if path:
                self.dataset[key] = path
                path_var.set(path)
            if key == "folder":
                channels = find_miniseed_channels(path)
                if channels:
                    self.dataset["channels"] = [
                        f"{loc}.{ch}" if loc else ch for loc, ch in channels
                    ]
                    self.channels_text.delete("1.0", tk.END)
                    self.channels_text.insert(
                        "1.0", "\n".join(self.dataset["channels"])
                    )

        path_var = tk.StringVar(value=self.dataset.get(key, ""))
        label = ttk.Label(self, text=label_text)
        label.grid(row=row, column=0, sticky="w")
        # No tooltip for folders/files
        ttk.Entry(self, textvariable=path_var, width=40).grid(
            row=row, column=1, sticky="w"
        )
        ttk.Button(self, text="Select", command=select_path).grid(
            row=row, column=2, sticky="w"
        )

    def build(self):
        row = 0
        # Folders/files: no tooltip, plain label
        self.build_path_selector(
            PARAM_LABELS.get(
                "folder",
                "folder"),
            "folder",
            row)
        row += 1
        self.build_path_selector(
            PARAM_LABELS.get("response", "response"), "response", row
        )
        row += 1
        # Action
        label = ttk.Label(
            self, text=PARAM_LABELS.get(
                "action", "Action") + ":")
        label.grid(row=row, column=0, sticky="w")
        ToolTip(label, PARAM_TOOLTIPS.get("action", ""))
        self.action_var = tk.StringVar(
            value=self.dataset.get(
                "action", "full"))
        action_combo = ttk.Combobox(
            self,
            textvariable=self.action_var,
            values=ACTIONS,
            state="readonly",
            width=15,
        )
        action_combo.grid(row=row, column=1, sticky="w")
        action_combo.bind("<<ComboboxSelected>>", self.update_action)
        row += 1

        # Timewindow
        label = ttk.Label(
            self, text=PARAM_LABELS.get("timewindow", "Timewindow") + " (s):"
        )
        label.grid(row=row, column=0, sticky="w")
        ToolTip(label, PARAM_TOOLTIPS.get("timewindow", ""))
        self.tw_var = tk.StringVar(
            value=str(
                self.dataset.get(
                    "timewindow",
                    3600)))
        tw_entry = ttk.Entry(self, textvariable=self.tw_var, width=10)
        tw_entry.grid(row=row, column=1, sticky="w")
        tw_entry.bind("<FocusOut>", self.update_timewindow)
        row += 1

        # Channels
        label = ttk.Label(self, text=PARAM_LABELS.get("channels", "channels"))
        label.grid(row=row, column=0, sticky="nw")
        ToolTip(label, PARAM_TOOLTIPS.get("channels", ""))
        self.channels_text = tk.Text(self, height=3, width=40)
        channels_val = self.dataset.get("channels", [])
        if isinstance(channels_val, list):
            self.channels_text.insert("1.0", "\n".join(channels_val))
        elif isinstance(channels_val, str):
            self.channels_text.insert("1.0", channels_val)
        self.channels_text.grid(row=row, column=1, columnspan=2, sticky="w")
        self.channels_text.bind("<FocusOut>", self.update_channels)
        row += 1

        # Plot options
        label = ttk.Label(self, text="Plot Options:")
        label.grid(row=row, column=0, sticky="w")
        row += 1

        for key in sorted(PLOT_KWARGS):
            current_val = self.dataset["plot_kwargs"].get(key)
            if key in BOOLEAN_KEYS:
                var = tk.BooleanVar(value=safe_bool(current_val))
                cb = ttk.Checkbutton(
                    self, text=PARAM_LABELS.get(key, key), variable=var
                )
                cb.grid(row=row, column=0, sticky="w")
                ToolTip(cb, PARAM_TOOLTIPS.get(key, ""))
                var.trace_add(
                    "write", partial(
                        self.update_plot_kwargs, key, var))
                self.plot_kwargs_vars[key] = var
            elif key == "cmap":
                current_val = str(current_val) if current_val else "pqlx"
                label = ttk.Label(
                    self, text=PARAM_LABELS.get(
                        "cmap", "cmap") + ":")
                label.grid(row=row, column=0, sticky="w")
                ToolTip(label, PARAM_TOOLTIPS.get("cmap", ""))
                var = tk.StringVar(value=current_val)
                combo = ttk.Combobox(
                    self,
                    textvariable=var,
                    values=CMAP_NAMES,
                    state="readonly",
                    width=30,
                )
                combo.grid(row=row, column=1, sticky="w")
                combo.bind(
                    "<<ComboboxSelected>>", partial(
                        self.update_plot_kwargs, key, var))
                self.plot_kwargs_vars[key] = var
            else:
                var = tk.StringVar(
                    value=str(current_val) if current_val is not None else ""
                )
                label = ttk.Label(self, text=PARAM_LABELS.get(key, key) + ":")
                label.grid(row=row, column=0, sticky="w")
                ToolTip(label, PARAM_TOOLTIPS.get(key, ""))
                ent = ttk.Entry(self, textvariable=var, width=30)
                ent.grid(row=row, column=1, sticky="w")
                ent.bind(
                    "<FocusOut>", partial(
                        self.update_plot_kwargs, key, var))
                self.plot_kwargs_vars[key] = var
            row += 1

        self.progress = ttk.Progressbar(self, maximum=100, mode="determinate")
        self.progress.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(
                5,
                0))
        row += 1
        self.status_label = ttk.Label(self, text="", foreground="gray")
        self.status_label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        # Group buttons in a frame
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Button(
            btn_frame,
            text="Run Dataset",
            command=self.run_this_dataset).pack(
            side="left",
            padx=(
                0,
                5))
        ttk.Button(
            btn_frame, text="Delete Dataset", command=self.delete_this_dataset
        ).pack(side="left", padx=(0, 5))
        ttk.Button(
            btn_frame,
            text="Duplicate Dataset",
            command=self.duplicate_this_dataset).pack(
            side="left",
            padx=(
                0,
                5))

    def update_timewindow(self, *_):
        try:
            self.dataset["timewindow"] = int(self.tw_var.get())
        except ValueError:
            self.dataset["timewindow"] = 3600

    def update_action(self, *_):
        val = self.action_var.get()
        if val in ACTIONS:
            self.dataset["action"] = val

    def update_channels(self, event=None):
        text = self.channels_text.get("1.0", "end").strip()
        self.dataset["channels"] = [
            line.strip() for line in text.splitlines() if line.strip()
        ]

    def update_plot_kwargs(self, key, var, *_):
        val = var.get()
        if key in BOOLEAN_KEYS:
            self.dataset["plot_kwargs"][key] = bool(val)
        elif key == "show_earthquakes":
            val = val.strip()
            if not val:
                self.dataset["plot_kwargs"][key] = None
            else:
                try:
                    # Accept comma or space separated values
                    parts = [
                        float(x.strip())
                        for x in val.replace(" ", ",").split(",")
                        if x.strip()
                    ]
                    if len(parts) == 1:
                        self.dataset["plot_kwargs"][key] = (parts[0],)
                    elif len(parts) >= 2:
                        self.dataset["plot_kwargs"][key] = tuple(parts[:2])
                    else:
                        self.dataset["plot_kwargs"][key] = None
                except Exception:
                    self.dataset["plot_kwargs"][key] = None
        else:
            try:
                if "," in val:
                    val = [float(x.strip()) for x in val.split(",")]
                else:
                    val = float(val)
            except ValueError:
                pass  # Keep as string
            self.dataset["plot_kwargs"][key] = val

    def run_this_dataset(self):
        self.status_label.config(text="Starting...", foreground="orange")
        self.progress["value"] = 0

        def update_progress(val, status):
            self.progress["value"] = val
            self.status_label.config(text=status)
            self.update_idletasks()

        tw = self.dataset.get("timewindow", 3600)
        try:
            process_dataset_visual(self.dataset, tw, update_progress)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", foreground="red")

    def delete_this_dataset(self):
        if self.delete_callback:
            self.delete_callback(self.index)

    def duplicate_this_dataset(self):
        if self.duplicate_callback:
            self.duplicate_callback(self.index)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPSD Plotter GUI")
        self.geometry("1000x700")
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        self.datasets = [copy.deepcopy(DEFAULT_DATASET)]
        self.selected_dataset_index = None
        self.build_menu()
        self.build_main()
        self.populate_datasets()

    def build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Config", command=self.new_config)
        filemenu.add_command(label="Load Config", command=self.load_config)
        filemenu.add_command(label="Save Config", command=self.save_config)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def build_main(self):
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.scroll = tk.Canvas(self.container)
        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.scroll.yview
        )
        self.hscrollbar = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.scroll.xview
        )
        self.scroll_frame = ttk.Frame(self.scroll)
        self.scroll_frame.bind(
            "<Configure>", lambda e: self.scroll.configure(
                scrollregion=self.scroll.bbox("all")), )
        self.scroll.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll.configure(
            yscrollcommand=self.scrollbar.set,
            xscrollcommand=self.hscrollbar.set)

        self.scroll.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.hscrollbar.pack(side="bottom", fill="x")

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)
        ttk.Button(
            controls,
            text="Add Dataset",
            command=self.add_dataset).pack(
            side="left")

    def select_dataset(self, index):
        self.selected_dataset_index = index
        self.populate_datasets()  # Refresh to update selection highlight

    def add_dataset(self):

        self.datasets.append(copy.deepcopy(DEFAULT_DATASET))
        self.populate_datasets()

    def delete_selected_dataset(self):
        if (
            self.selected_dataset_index is not None
            and 0 <= self.selected_dataset_index < len(self.datasets)
        ):
            del self.datasets[self.selected_dataset_index]
            self.selected_dataset_index = None
            self.populate_datasets()

    def populate_datasets(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        for idx, ds in enumerate(self.datasets):
            ds.setdefault("plot_kwargs", {})
            frame = DatasetFrame(
                self.scroll_frame,
                ds,
                idx,
                run_callback=self.run_dataset,
                delete_callback=self.delete_dataset,
                duplicate_callback=self.duplicate_dataset,
            )
            row, col = divmod(idx, 3)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        for i in range(3):
            self.scroll_frame.columnconfigure(i, weight=1)

    def run_dataset(self, index):
        ds = self.datasets[index]
        process_dataset_visual(ds)  # Only pass ds, not self

    def delete_dataset(self, index):
        if 0 <= index < len(self.datasets):
            del self.datasets[index]
            self.populate_datasets()

    def duplicate_dataset(self, index):
        if 0 <= index < len(self.datasets):

            self.datasets.insert(
                index + 1,
                copy.deepcopy(
                    self.datasets[index]))
            self.populate_datasets()

    def load_config(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if not filepath:
            return
        with open(filepath) as f:
            config = yaml.safe_load(f)
        self.datasets = config.get("datasets", [])
        for ds in self.datasets:
            # Move top-level plot kwargs into plot_kwargs dict
            plot_kwargs = ds.get("plot_kwargs", {})
            for key in list(ds.keys()):
                if key in PLOT_KWARGS:
                    plot_kwargs[key] = ds.pop(key)
            ds["plot_kwargs"] = normalize_plot_kwargs(plot_kwargs)

        self.populate_datasets()

    def save_config(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml")],
        )
        if not filepath:
            return
        for ds in self.datasets:
            if ds.get("plot_kwargs", {}).get("cmap") == pqlx:
                ds["plot_kwargs"]["cmap"] = "pqlx"
        config = {"datasets": make_yaml_safe(self.datasets)}
        with open(filepath, "w") as f:
            yaml.dump(config, f)

    def new_config(self):
        if self.datasets:
            confirm = messagebox.askyesno(
                "New Configuration",
                "This will discard the current configuration.\nContinue?")
        else:
            confirm = True
        if not confirm:
            return
        self.datasets = [copy.deepcopy(DEFAULT_DATASET)]
        self.populate_datasets()


if __name__ == "__main__":
    app = App()
    app.mainloop()
