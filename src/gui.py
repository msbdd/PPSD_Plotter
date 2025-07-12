import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import yaml
from pathlib import Path
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import get_named_colors_mapping
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from obspy import read
import matplotlib
from functools import partial
import os
import sys
import platform
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from ppsd_plotter_aux import calculate_ppsd_worker, load_inventory, \
                            find_miniseed_channels, find_miniseed, \
                            calculate_noise_line
from localization_dicts import ALL_LABELS, ALL_SOFTWARE_LABELS, ALL_TOOLTIPS

matplotlib.use("TkAgg")
CURRENT_LANG = "EN"
PARAM_LABELS = ALL_LABELS[CURRENT_LANG]
PARAM_TOOLTIPS = ALL_TOOLTIPS[CURRENT_LANG]
SOFTWARE_LABELS = ALL_SOFTWARE_LABELS[CURRENT_LANG]
CMAP_NAMES = ["pqlx"] + sorted(colormaps)
COLOR_NAMES = sorted(
    name for name in get_named_colors_mapping().keys() if " " not in name
)

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
    "timewindow": 3600,
    "plot_kwargs": DEFAULT_PLOT_KWARGS.copy(),
}

ACTIONS = ["plot", "calculate", "full", "convert"]


class ProgressReporter:
    def __init__(self, total_tasks, callback=None):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.callback = callback

    def update(self, label=None):
        self.completed_tasks += 1
        percent = int((self.completed_tasks / self.total_tasks) * 100)
        if self.callback:
            self.callback(
                percent, label or f"{self.completed_tasks}/{self.total_tasks}"
            )


def chunk_files(tasks, max_workers):
    chunk_count = min(len(tasks), max_workers)
    return [tasks[i::chunk_count] for i in range(chunk_count)]


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')
        )
    return os.path.join(base_path, relative_path)


def parse_channel(ch_str):
    parts = ch_str.split(".", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


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


def format_plot_kwargs_for_display(plot_kwargs, param_labels_dict):
    lines = []
    for key, value in plot_kwargs.items():
        label = param_labels_dict.get(key, key)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)


def calculate_ppsd(
        folder, inv, tw, channel_list, callback=None, max_workers=None
        ):
    folder = Path(folder)
    files = list(folder.rglob("*"))
    files = [
        f for f in files if f.suffix.lower() in [".mseed", ".miniseed", ".msd"]
        ]

    if not files:
        if callback:
            callback(0, "No data files found.")
        return

    channels_set = set()
    for ch in channel_list:
        parts = ch.strip().split(".")
        if len(parts) == 2:
            loc, chan = parts
        else:
            loc, chan = None, parts[0]
        channels_set.add((loc, chan))

    job_list = []
    for file in files:
        try:
            st = read(str(file))
        except Exception as e:
            print(f"Failed reading {file.name}: {e}")
            continue
        for loc, chan in channels_set:
            traces = st.select(channel=chan, location=loc if loc else "")
            if traces:
                job_list.append((file, loc, chan))

    if not job_list:
        if callback:
            callback(0, "No matching traces found.")
        return

    cpu_count = max_workers or max(1, os.cpu_count() - 2)
    chunks = chunk_files(job_list, cpu_count)

    progress_total = len(chunks)
    progress_done = 0

    def update_progress():
        nonlocal progress_done
        progress_done += 1
        if callback:
            pct = int((progress_done / progress_total) * 100)
            callback(pct, f"Processing {progress_done}/{progress_total}")

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [
            executor.submit(calculate_ppsd_worker, chunk, inv, tw, folder)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            future.result()
            update_progress()


def process_dataset_visual(ds, progress_update_callback):
    folder = ds.get("folder", "")
    resp_file = ds.get("response", "")
    channels = ds.get("channels", [])
    action = str(ds.get("action", "full"))
    inv = load_inventory(resp_file)

    if not inv:
        progress_update_callback(0, "Failed: inventory load")
        return

    if not channels:
        progress_update_callback(0, "No channels defined")
        return

    parsed_channels = []
    for ch_str in channels:
        loc_code, channel = parse_channel(ch_str)
        parsed_channels.append((loc_code, channel))

    plot_kwargs = normalize_plot_kwargs(ds.get("plot_kwargs", {}))

    if action in ["calculate", "full"]:
        calculate_ppsd(
            folder=folder,
            inv=resp_file,
            tw=int(ds.get("timewindow", 3600)),
            channel_list=channels,
            callback=progress_update_callback,
        )

    for i, (loc_code, channel) in enumerate(parsed_channels):
        progress = int((i / len(parsed_channels)) * 100)
        ch_label = f"{loc_code}.{channel}" if loc_code else channel
        if loc_code:
            npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
        else:
            npzfolder = Path(folder) / f"npz_{channel}"
        if action in ["plot", "full"]:
            sample = find_miniseed(folder, channel, loc_code)
            if sample:
                plot_ppsd_interactive(
                    sample,
                    channel,
                    loc_code,
                    inv,
                    npzfolder,
                    int(ds.get("timewindow", 3600)),
                    plot_kwargs.copy(),
                    custom_noise_line=ds.get("custom_noise_line")
                )
            else:
                progress_update_callback(progress, f"No data for {ch_label}")

        if action == "convert":
            convert_npz_to_text(npzfolder)

    progress_update_callback(100, "Done")


def plot_ppsd_interactive(
    sampledata, channel, location, inv,
    npzfolder, tw, plot_kwargs=None,
    custom_noise_line=None
):
    if plot_kwargs is None:
        plot_kwargs = {}
    cmap_name_or_obj = plot_kwargs.pop("cmap", "pqlx")
    if isinstance(cmap_name_or_obj, str):
        if cmap_name_or_obj == "pqlx":
            cmap = pqlx
        else:
            cmap = colormaps.get(cmap_name_or_obj, "viridis")
    else:
        cmap = cmap_name_or_obj
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
    if custom_noise_line:
        try:
            amp = float(custom_noise_line.get("amplitude", 1.0))
            f1, f2 = map(
                float, custom_noise_line.get("freq_range", [1.0, 10.0])
                )
            color = custom_noise_line.get("color", "orange")

            freq1, freq2, db = calculate_noise_line(amp, (f1, f2))
            if not plot_kwargs.get("xaxis_frequency"):
                freq1 = 1/freq1
                freq2 = 1/freq2
            ax = fig.axes[0]
            ax.plot([freq1, freq2], [db, db],
                    color=color,
                    lw=2,
                    linestyle='--',
                    label=f"Noise {amp:.6g} mg")
            leg = ax.legend()
            leg.set_draggable(True)
        except Exception as e:
            print(f"Failed to plot custom noise line: {e}")
    fig.canvas.manager.set_window_title(f"PPSD Plot {trace.id}")
    fig.show()


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
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("dataset")
        super().__init__(
            parent,
            text=f"{textlabel} {index+1}",
            padding=5
        )
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
        ttk.Entry(self, textvariable=path_var, width=40).grid(
            row=row, column=1, sticky="w"
        )
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("select")
        ttk.Button(self, text=textlabel, command=select_path).grid(
            row=row, column=2, sticky="w"
        )

    def build(self):
        row = 0
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

        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("options")
        label = ttk.Label(self, text=textlabel)
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

        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("custom_noise")
        label = ttk.Label(self, text=textlabel)
        label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.custom_noise_vars = {
            "amplitude": tk.StringVar(value=""),
            "freq_range": tk.StringVar(value=""),
            "color": tk.StringVar(value="")
        }

        c = self.dataset.get("custom_noise_line")
        if isinstance(c, dict):
            self.custom_noise_vars["amplitude"].set(
                str(c.get("amplitude", ""))
                )
            self.custom_noise_vars["freq_range"].set(
                ", ".join(map(str, c.get("freq_range", [])))
            )
            self.custom_noise_vars["color"].set(c.get("color", ""))

        for field, var in self.custom_noise_vars.items():
            label_text = ALL_SOFTWARE_LABELS[CURRENT_LANG].get(field)
            label = ttk.Label(
                self, text=label_text
                )
            label.grid(row=row, column=0, sticky="w")

            if field == "color":
                combo = ttk.Combobox(
                    self, textvariable=var, values=COLOR_NAMES, width=30
                    )
                combo.grid(row=row, column=1, sticky="w")
                combo.bind(
                    "<<ComboboxSelected>>", lambda e,
                    f=field: self.update_custom_noise_field(f)
                    )
                combo.bind(
                    "<FocusOut>", lambda e,
                    f=field: self.update_custom_noise_field(f)
                    )
            else:
                entry = ttk.Entry(self, textvariable=var, width=30)
                entry.grid(row=row, column=1, sticky="w")
                entry.bind(
                    "<FocusOut>", lambda e,
                    f=field: self.update_custom_noise_field(f)
                    )

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
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("run")
        ttk.Button(
            btn_frame,
            text=textlabel,
            command=self.run_this_dataset).pack(
            side="left",
            padx=(
                0,
                5))
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("delete")
        ttk.Button(
            btn_frame, text=textlabel, command=self.delete_this_dataset
        ).pack(side="left", padx=(0, 5))
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("duplicate")
        ttk.Button(
            btn_frame,
            text=textlabel,
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
            clean_val = safe_bool(val)
            self.dataset["plot_kwargs"][key] = clean_val
            return

        elif key == "cmap":
            if val:
                self.dataset["plot_kwargs"][key] = val
            else:
                self.dataset["plot_kwargs"][key] = None
            return

        elif key == "show_earthquakes":
            if not val:
                self.dataset["plot_kwargs"][key] = None
                var.set("")  # clear
                return
            try:
                parts = [
                    float(x.strip())
                    for x in val.replace(" ", ",").split(",")
                    if x.strip()
                ]
                if len(parts) == 1:
                    parsed_val = (parts[0],)
                else:
                    parsed_val = tuple(parts[:2])
                self.dataset["plot_kwargs"][key] = parsed_val
                var.set(", ".join(map(str, parsed_val)))
            except Exception:
                self.dataset["plot_kwargs"][key] = None
                var.set("")
            return
        elif key == "percentiles":
            if not val:
                self.dataset["plot_kwargs"][key] = []
                var.set("")
                return
            try:
                parts = [
                    float(x.strip())
                    for x in val.replace(" ", ",").split(",")
                    if x.strip()
                ]
                self.dataset["plot_kwargs"][key] = parts
                var.set(", ".join(map(str, parts)))
            except Exception:
                self.dataset["plot_kwargs"][key] = []
                var.set("")
            return
        try:
            if "," in val:
                parts = [float(x.strip()) for x in val.split(",") if x.strip()]
                self.dataset["plot_kwargs"][key] = parts
                var.set(", ".join(map(str, parts)))
            else:
                number = float(val)
                self.dataset["plot_kwargs"][key] = number
                var.set(str(number))
        except Exception:
            self.dataset["plot_kwargs"][key] = None
            var.set("")

    def update_custom_noise_field(self, field):
        if not isinstance(self.dataset.get("custom_noise_line"), dict):
            self.dataset["custom_noise_line"] = {}

        line = self.dataset["custom_noise_line"]
        value = self.custom_noise_vars[field].get().strip()

        if field == "amplitude":
            try:
                line["amplitude"] = float(value)
            except ValueError:
                line.pop("amplitude", None)
                self.custom_noise_vars["amplitude"].set("")

        elif field == "freq_range":
            try:
                parts = [
                    float(x.strip())
                    for x in value.replace(" ", ",").split(",")
                    if x.strip()
                ]
                if len(parts) == 2:
                    line["freq_range"] = tuple(parts)
                else:
                    raise ValueError
            except Exception:
                line.pop("freq_range", None)
                self.custom_noise_vars["freq_range"].set("")

        elif field == "color":
            if value.lower() in COLOR_NAMES:
                line["color"] = value.lower()
            else:
                line.pop("color", None)
                self.custom_noise_vars["color"].set("")

        if (not line.get("amplitude") and
                not line.get("freq_range") and
                not line.get("color")):
            self.dataset["custom_noise_line"] = None

    def run_this_dataset(self):
        self.status_label.config(text="Starting...", foreground="orange")
        self.progress["value"] = 0

        def update_progress(val, status):
            self.progress["value"] = val
            self.status_label.config(text=status)
            self.update_idletasks()

        try:
            process_dataset_visual(self.dataset, update_progress)
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
        if platform.system() == "Windows":
            self.state('zoomed')
            icon_path = resource_path("resources/icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        elif platform.system() == "Linux":
            self.attributes('-zoomed', True)
            icon_path = resource_path("resources/icon.png")
            if os.path.exists(icon_path):
                icon_image = tk.PhotoImage(file=icon_path)
                self.iconphoto(False, icon_image)
        self.datasets = [copy.deepcopy(DEFAULT_DATASET)]
        self.selected_dataset_index = None
        self.build_menu()
        self.build_main()
        self.populate_datasets()

    def build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("new_config")
        filemenu.add_command(
            label=textlabel,
            command=self.new_config)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("load_config")
        filemenu.add_command(
            label=textlabel,
            command=self.load_config)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("save_config")
        filemenu.add_command(
            label=textlabel,
            command=self.save_config)
        filemenu.add_separator()
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("exit")
        filemenu.add_command(label=textlabel, command=self.quit)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("file")
        menubar.add_cascade(label=textlabel, menu=filemenu)
        self.config(menu=menubar)
        lang_menu = tk.Menu(menubar, tearoff=0)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("language")
        menubar.add_cascade(label=textlabel, menu=lang_menu)
        for lang in ALL_LABELS.keys():
            lang_menu.add_command(
                label=lang, command=lambda la=lang: self.set_language(la))

    def build_main(self):
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.scroll = tk.Canvas(self.container, borderwidth=0)
        self.scroll.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.scroll.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.hscrollbar = ttk.Scrollbar(
            self, orient="horizontal", command=self.scroll.xview)
        self.hscrollbar.pack(side="bottom", fill="x")

        self.scroll.configure(yscrollcommand=self.scrollbar.set,
                              xscrollcommand=self.hscrollbar.set)

        self.scroll_frame = ttk.Frame(self.scroll)
        self.scroll_window = self.scroll.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )

        def update_scroll_region(event):
            self.scroll.configure(scrollregion=self.scroll.bbox("all"))

        self.scroll_frame.bind("<Configure>", update_scroll_region)

        def _on_mousewheel(event):
            self.scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_mac(event):
            self.scroll.yview_scroll(int(-1 * (event.delta)), "units")

        def _bind_to_mousewheel(event):
            if sys.platform == "darwin":
                self.scroll.bind_all("<MouseWheel>", _on_mousewheel_mac)
            else:
                self.scroll.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.scroll.unbind_all("<MouseWheel>")

        self.scroll_frame.bind("<Enter>", _bind_to_mousewheel)
        self.scroll_frame.bind("<Leave>", _unbind_from_mousewheel)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)
        textlabel = ALL_SOFTWARE_LABELS[CURRENT_LANG].get("add", "Add Dataset")
        ttk.Button(controls, text=textlabel,
                   command=self.add_dataset).pack(side="left")
        ttk.Button(
            controls,
            text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get("run_all"),
            command=self.open_run_all_dialog
        ).pack(side="left", padx=(10, 0))

    def select_dataset(self, index):
        self.selected_dataset_index = index
        self.populate_datasets()

    def set_language(self, lang_code):
        global CURRENT_LANG, PARAM_LABELS, PARAM_TOOLTIPS, SOFTWARE_LABELS
        CURRENT_LANG = lang_code
        PARAM_LABELS = ALL_LABELS[lang_code]
        PARAM_TOOLTIPS = ALL_TOOLTIPS.get(lang_code, {})
        SOFTWARE_LABELS = ALL_SOFTWARE_LABELS[lang_code]
        self.rebuild_gui()

    def rebuild_gui(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.build_menu()
        self.build_main()
        self.populate_datasets()

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
        process_dataset_visual(ds)

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

    def open_run_all_dialog(self):
        GRID_PAD = {"padx": 10, "pady": 5}
        dialog = tk.Toplevel(self)
        dialog.title(ALL_SOFTWARE_LABELS[CURRENT_LANG].get("run_all_save"))
        dialog.geometry("600x400")
        dialog.grab_set()
        dialog.focus_force()
        if platform.system() == "Windows":
            icon_path = resource_path("resources/icon.ico")
            if os.path.exists(icon_path):
                dialog.iconbitmap(icon_path)
        elif platform.system() == "Linux":
            icon_path = resource_path("resources/icon.png")
            if os.path.exists(icon_path):
                icon_image = tk.PhotoImage(file=icon_path)
                dialog.iconphoto(False, icon_image)
        ttk.Label(dialog, text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get(
            "output_folder")).grid(row=0, column=0, sticky="w", **GRID_PAD)
        out_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=out_var, width=40).grid(
            row=0, column=1, sticky="w", **GRID_PAD)

        def browse_folder():
            selected = filedialog.askdirectory(parent=dialog)
            if selected:
                out_var.set(selected)
                dialog.focus_force()

        ttk.Button(dialog, text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get(
            "browse"), command=browse_folder).grid(row=0, column=2, **GRID_PAD)

        ttk.Label(dialog, text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get(
            "img_size")).grid(row=1, column=0, sticky="w", **GRID_PAD)
        size_var = tk.StringVar(value="12,8")
        ttk.Entry(dialog, textvariable=size_var).grid(
            row=1, column=1, sticky="w", **GRID_PAD)

        ttk.Label(dialog, text="DPI:").grid(
            row=2, column=0, sticky="w", **GRID_PAD)
        dpi_var = tk.StringVar(value="300")
        ttk.Entry(dialog, textvariable=dpi_var).grid(
            row=2, column=1, sticky="w", **GRID_PAD)

        ttk.Label(
            dialog, text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get("progress")
            ).grid(
            row=3, column=0, columnspan=2, sticky="w", **GRID_PAD
            )

        progress_bars = []
        status_labels = []

        for i in range(len(self.datasets)):
            ttk.Label(
                dialog,
                text=(
                    f"{ALL_SOFTWARE_LABELS[CURRENT_LANG].get('dataset')} {i+1}"
                )
                ).grid(
                    row=4 + i * 2, column=0, sticky="w", **GRID_PAD
                )
            bar = ttk.Progressbar(dialog, maximum=100)
            bar.grid(row=4 + i * 2, column=1,
                     columnspan=2, sticky="ew", **GRID_PAD)
            label = ttk.Label(dialog, text="", foreground="gray", )
            label.grid(row=5 + i * 2, column=0,
                       columnspan=3, sticky="w", **GRID_PAD)
            progress_bars.append(bar)
            status_labels.append(label)

        def run_all():
            try:
                width, height = map(float, size_var.get().split(","))
                dpi = int(dpi_var.get())
                output_folder = out_var.get()

                for i, dataset in enumerate(self.datasets):
                    def local_callback(
                            pct, msg, b=progress_bars[i], lbl=status_labels[i]
                            ):
                        b["value"] = pct
                        lbl["text"] = msg
                        lbl.update()

                    self.run_single_dataset_to_file(
                        dataset, output_folder,
                        (width, height), dpi, local_callback
                        )
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {e}")
            dialog.destroy()

        ttk.Button(dialog, text=ALL_SOFTWARE_LABELS[CURRENT_LANG].get("run"),
                   command=run_all).grid(
            row=5 + 2 * len(self.datasets), column=1, pady=10
        )

    def run_single_dataset_to_file(
            self, ds, output_folder, figsize, dpi, callback
            ):

        folder = ds.get("folder", "")
        resp_file = ds.get("response", "")
        channels = ds.get("channels", [])
        action = str(ds.get("action", "full")).lower()
        matplotlib.use("Agg")

        inv = load_inventory(resp_file)

        if not inv or not channels:
            callback(0, "Invalid dataset")
            return

        parsed_channels = [parse_channel(ch) for ch in channels]
        plot_kwargs = normalize_plot_kwargs(ds.get("plot_kwargs", {}))
        if action in ["calculate", "full"]:
            calculate_ppsd(
                folder=folder,
                inv=resp_file,
                tw=int(ds.get("timewindow", 3600)),
                channel_list=channels,
                callback=callback,
            )

        if action in ["plot", "full"]:
            cmap_name_or_obj = plot_kwargs.pop("cmap", "pqlx")
            if isinstance(cmap_name_or_obj, str):
                if cmap_name_or_obj == "pqlx":
                    cmap = pqlx
                else:
                    cmap = colormaps.get(
                        cmap_name_or_obj, "viridis")
            else:
                cmap = cmap_name_or_obj
            for i, (loc_code, channel) in enumerate(parsed_channels):
                ch_label = f"{loc_code}.{channel}" if loc_code else channel
                if loc_code:
                    npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
                else:
                    npzfolder = Path(folder) / f"npz_{channel}"
                sample = find_miniseed(folder, channel, loc_code)
                if sample:
                    try:
                        st = read(sample)
                        tr = st.select(channel=channel, location=loc_code)[0]
                        ppsd = PPSD(
                            tr.stats, inv,
                            ppsd_length=int(ds.get("timewindow", 3600))
                            )

                        for file in Path(npzfolder).glob("*.npz"):
                            ppsd.add_npz(str(file))

                        fig = ppsd.plot(cmap=cmap, show=False, **plot_kwargs)
                        custom_noise_line = ds.get("custom_noise_line")
                        if isinstance(custom_noise_line, dict):
                            amp = float(custom_noise_line.get(
                                "amplitude", 0.001)
                                )
                            freq_range = custom_noise_line.get(
                                "freq_range", []
                                )
                            color = custom_noise_line.get("color", "red")

                            if len(freq_range) == 2 and amp > 0:
                                freq1, freq2, db = calculate_noise_line(
                                    amp, tuple(freq_range)
                                    )

                                if not plot_kwargs.get(
                                    "xaxis_frequency", True
                                        ):
                                    freq1 = 1.0 / freq1 if freq1 != 0 else 0
                                    freq2 = 1.0 / freq2 if freq2 != 0 else 0

                                ax = fig.axes[0]
                                ax.plot([freq1, freq2], [db, db],
                                        color=color,
                                        lw=2,
                                        linestyle='--',
                                        label=f"Noise {amp:.6g} mg")
                                ax.legend()

                        figfile = Path(output_folder) / f"{tr.id}.png"
                        fig.set_size_inches(figsize)
                        fig.savefig(figfile, dpi=dpi)
                        plt.close(fig)

                        pct = int((i + 1) / len(parsed_channels) * 100)
                        callback(pct, f"Saved: {tr.id}")
                    except Exception as e:
                        callback(0, f"Plot failed: {e}")
                else:
                    callback(0, f"No data for {ch_label}")

        if action == "convert":
            for loc_code, channel in parsed_channels:
                if loc_code:
                    npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
                else:
                    npzfolder = Path(folder) / f"npz_{channel}"
                try:
                    convert_npz_to_text(npzfolder)
                    callback(100, f"Converted {channel}")
                except Exception as e:
                    callback(0, f"Convert failed: {e}")

        if action not in ["calculate", "plot", "full", "convert"]:
            callback(0, f"Unknown action: {action}")
        matplotlib.use("TkAgg")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
