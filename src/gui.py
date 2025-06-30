import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import yaml
from pathlib import Path
from obspy import read_inventory
from obspy import read
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from PPSD_plotter import process_dataset, load_config, \
    parse_channel, find_miniseed
import matplotlib
matplotlib.use("TkAgg")


def plot_ppsd_interactive(
        sampledata, channel, location, inv, npzfolder,
        tw, plot_kwargs=None
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

    for file in Path(npzfolder).glob("*.npz"):
        try:
            ppsd.add_npz(str(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    cmap = plot_kwargs.pop("cmap", pqlx)
    figsize = plot_kwargs.pop("figsize", (12, 6))

    fig = ppsd.plot(cmap=cmap, show=False, **plot_kwargs)
    fig.set_size_inches(*figsize)
    fig.canvas.manager.set_window_title(f"PPSD Plot {trace.id}")
    fig.show()


class DatasetFrame(ttk.LabelFrame):
    def __init__(self, parent, dataset, index, tw):
        super().__init__(parent, text=f"Dataset {index+1}")
        self.dataset = dataset
        self.tw = tw
        self.figures = []
        self.build()

    def build(self):
        ttk.Label(
            self, text=f"Folder: {self.dataset['folder']}"
            ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self, text=f"Channels: {', '.join(self.dataset['channels'])}"
            ).grid(row=1, column=0, sticky="w")
        ttk.Button(
            self, text="Run Task", command=self.run_task
            ).grid(row=2, column=0, sticky="w")
        self.plot_canvas = tk.Canvas(self, height=200, width=400)
        self.plot_canvas.grid(row=3, column=0, pady=10)

    def run_task(self):
        try:
            process_dataset(self.dataset, self.tw)
            self.show_interactive_plots()
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")

    def show_interactive_plots(self):
        folder = self.dataset["folder"]
        inv = read_inventory(self.dataset["response"])
        for ch_str in self.dataset["channels"]:
            loc_code, channel = parse_channel(ch_str)
            if loc_code:
                npzfolder = Path(folder) / f"npz_{loc_code}_{channel}"
            else:
                npzfolder = Path(folder) / f"npz_{channel}"
            sample = find_miniseed(folder, channel, loc_code)
            plot_ppsd_interactive(
                sampledata=sample,
                channel=channel,
                location=loc_code,
                inv=inv,
                npzfolder=npzfolder,
                tw=self.tw,
                plot_kwargs={}
            )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPSD Plotter GUI")
        self.geometry("800x600")
        self.datasets = []
        self.tw = 3600  # Default fallback
        self.build_menu()
        self.build_main()

    def build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Config", command=self.load_config)
        filemenu.add_command(label="Save Config", command=self.save_config)
        filemenu.add_command(label="New Config", command=self.new_config)
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
        self.scroll_frame = ttk.Frame(self.scroll)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.scroll.configure(
                scrollregion=self.scroll.bbox("all")
                )
        )
        self.scroll.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
            )
        self.scroll.configure(yscrollcommand=self.scrollbar.set)

        self.scroll.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def populate_datasets(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        for idx, ds in enumerate(self.datasets):
            frame = DatasetFrame(self.scroll_frame, ds, idx, self.tw)
            frame.pack(fill="x", padx=10, pady=5)

    def load_config(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml *.yml")]
            )
        if not filepath:
            return
        config = load_config(filepath)
        self.datasets = config.get("datasets", [])
        self.tw = config.get("timewindow", 3600)
        self.populate_datasets()

    def save_config(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".yaml")
        if not filepath:
            return
        config = {"timewindow": self.tw, "datasets": self.datasets}
        with open(filepath, "w") as f:
            yaml.dump(config, f)

    def new_config(self):
        self.datasets = []
        self.populate_datasets()


if __name__ == "__main__":
    app = App()
    app.mainloop()
