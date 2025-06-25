# PPSD Plotting Script
~~<sup>(Because I am tired to always re-write the same code when I do need to plot PPSD)</sup>~~

This script automates the calculation, plotting, and export of Power Spectral Density (PPSD) data from seismic waveform files using [ObsPy](https://docs.obspy.org). It is designed to process many datasets using a simple YAML configuration and parallel execution.

## TODO:

- Better threading
- Additional configuration options

---

## Requirements

- Python 3.8.10+
- ObsPy
- PyYAML
- tqdm

## Installation

1) Create a new venv
2) Install dependencies:

```
pip install obspy pyyaml tqdm
```

---

## Configuration File: `config.yaml`

The script uses a YAML file to define how each dataset is processed.

```
timewindow: 3600           # PPSD time window in seconds
num_workers: 4             # Number of jobs to run in parallel

datasets:
  - folder: "data/st01"              # Path to waveform files (.mseed, .miniseed, .msd)
    response: "responses/ST01.xml"   # Station response file (.xml, .dataless)
    channels: ["HHZ", "HHE"]         # List of channels to process
    action: full                     # Processing action (see below)
    output_folder: "outputs/st01"    # Folder to save output plots

  - folder: "data/st02"
    response: "responses/ST02.xml"
    channels: ["BHZ"]
    action: calculate
    output_folder: "outputs/st02"
```

---

## Action Codes

Each dataset block includes an `action` value that controls what processing is performed:

```
calculate → Calculate PPSD and save as .npz files
plot → Plot existing .npz files as .png
full → Calculate and plot (equivalent to 1 + 2)
convert → Convert .npz files to a CSV text format
```

---

## Running the Script

Once configured, run the script using:

```
python PPSD_Plotter.py config.yaml
```

---

## Output Structure

Depending on the action used, the script generates the following output:

- `.npz` files saved to:
  ```
  <folder>/npz_<channel>/
  ```

- `.png` plots saved to:
  ```
  <output_folder>/<trace.id>.png
  ```

- `.csv` exported data (for action "convert") saved to:
  ```
  <folder>/npz_<channel>_text/export.csv
  ```


## Notes

- The waveform file extensions supported are:
  ```
  .mseed, .miniseed, .msd
  ```

- Each dataset must include:
  - One waveform folder
  - One response file
  - One or more channel codes

- Channels are processed independently, and multiple stations can be run in parallel.

---

## Contact

Please feel free to report issues or submit improvements!

---
