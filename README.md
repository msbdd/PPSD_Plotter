# PPSD Plotting Script

[![PPSD_Plotter Test Linux](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_Linux.yml/badge.svg)](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_Linux.yml)
[![PPSD_Plotter Test Windows](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_Windows.yml/badge.svg)](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_Windows.yml)
[![PPSD_Plotter Test MacOS](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_MacOS.yml/badge.svg)](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Test_MacOS.yml)
<br>
![Python versions:](https://img.shields.io/badge/python-3.8_%7C_3.9_%7C_3.10_%7C_3.11_%7C_3.12%7C_3.13-blue?)
<br>
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-brightgreen.svg)
![Code style: flake8](https://img.shields.io/badge/Code%20style-flake8-brightgreen)

[![PPSD_Plotter Build Windows](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Distribute_Windows.yml/badge.svg)](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Distribute_Windows.yml)

~~<sup>(Because I am tired to always re-write the same code when I do need to plot PPSD)</sup>~~

This script automates the calculation, plotting, and export of Power Spectral Density (PPSD) data from seismic waveform files using [ObsPy](https://docs.obspy.org). It is designed to process many datasets using a simple YAML configuration and parallel execution.

## TODO:

- Better threading
- GUI
- Wildcard support

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

The script uses a YAML file to define how each dataset is processed. <br> An example configuration is provided in the ```example``` folder.<br>
You can pass additional plotting parameters to the ```PPSD.plot()``` function from ObsPy.
For the full list of supported options, please refer to the [ObsPy documentation](https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.plot.html)

```
# 3600 should be used as a default; 60 is used for a small example dataset
timewindow: 600
num_workers: 2

datasets:

  - folder: "example/IU.ANMO..D"
    response: "example/IU_ANMO_RESP.xml"
    channels: ["BHZ", "BH1", "BH2"]
    action: full
    output_folder: "example/result"

  - folder: "example/IU.GRFO..D"
    response: "example/IU_GRFO_RESP.xml"
    channels: ["00.BHZ", "00.BH1", "00.BH2"]
    action: full
    output_folder: "example/result"
    figsize: [6, 6]
    show_mean: yes
    grid: no
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
  <folder>/npz_<location>_<channel>/
  ```

- `.png` plots saved to:
  ```
  <output_folder>/<trace.id>.png
  ```

- `.csv` exported data (for action "convert") saved to:
  ```
  <folder>/npz_<location>_<channel>_text/export.csv
  ```

## Example Data Acknowledgment

A simple example is provided in the example folder.
This example uses the seismic data from the [IU Network](https://www.fdsn.org/networks/detail/IU/)

The original data was collected by the **Albuquerque Seismological Laboratory / USGS** and distributed via the **International Federation of Digital Seismograph Networks (FDSN)**:

> Albuquerque Seismological Laboratory/USGS. (1988). *Global Seismograph Network (GSN - IRIS/USGS)* [Dataset]. International Federation of Digital Seismograph Networks. [https://doi.org/10.7914/SN/IU](https://doi.org/10.7914/SN/IU)

Seismic waveform and station metadata were accessed using the [EarthScope FDSN Web Services](https://service.iris.edu/), which provide public access to seismic data managed by the IRIS Data Management Center.

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
