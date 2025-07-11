# PPSD Plotting Utility

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-brightgreen.svg)
![Code style: flake8](https://img.shields.io/badge/Code%20style-flake8-brightgreen)

[![PPSD_Plotter Build Windows](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Distribute_Windows.yml/badge.svg)](https://github.com/msbdd/PPSD_Plotter/actions/workflows/Distribute_Windows.yml)

<sup>There is a saying: GUI makes simple tasks simpler and more convenient, while CLI makes complex tasks possible
<br>
This is primary aimed for solving a rather simple task, but without writing even simple scripts at the end user side.
</sup>

This utility automates the calculation, plotting, and export of Power Spectral Density (PPSD) data from seismic waveform files using [ObsPy](https://docs.obspy.org). It is designed to process many datasets using a simple YAML configuration and parallel execution and has a simple tkinter-based GUI.

## TODO:

- Custom plotting function and additional plotting parameters (for example, day/night or a custom noise level over a freq range)
- Wildcard support
- Linux building (?)

---

## Requirements

- Python 3.8.10+
- ObsPy
- PyYAML
- tqdm

## Installation

If you are on Windows and don't want to handle any python-related installations, please use the provided binary.

If you are on Linux/MacOS or want to install on Windows using source code you could:

1) Create a new venv
2) Install dependencies:

```
pip install -r requirements.txt
```
3) Run 
```
python src\gui.py
```
---

## Configuration File: `config.yaml`

The utility uses a YAML file to define how each dataset is processed. <br> An example configuration is provided in the ```example``` folder.<br>
You can pass additional plotting parameters to the ```PPSD.plot()``` function from ObsPy.
For the full list of supported options, please refer to the [ObsPy documentation](https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.plot.html)


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
