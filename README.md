# Thunderbird Data Analysis

This repository includes code for processing, analysing, and visualising data from the Thunderbird reactor. The primary purpose is to determine neutron count rates from EJ309 data, and process reactor data for comparison with count rates.

## Installation

Clone this repository:

```bash
git clone git@github.com:berlinguette/thunderbird_psd.git
```

Create a virtual environment:

```bash
cd thunderbird_psd
python3 -m venv .venv --prompt=tbird_psd
```

Activate your virtual environment:

- On Windows:

```cmd
.\.venv\Scripts\activate
```

- On Linux/MacOS:

```bash
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

Set up experimental data folders:
- Find a suitable main folder for all neutron data.
- In that folder, make 2 subfolders:
  - `2-Converted_Data`
  - `3-Output`

Create a `dot_env.py` file in the `data_processing` folder based on `dot_env.py.example`.

Set up notebook cleaning filters:

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.smudge cat
git config filter.strip-notebook-output.required true
git add --renormalize .
```

Note: VS Code Git integration will not work properly once these filters are set up.
Git command line should be used instead.

## Usage

TODO

Open Jupyter Lab

```bash
jupyter lab
```

Notebooks are stored in `active_notebooks`
Notebooks can typically be run using the `Run All` command
The `Reactor Data Time Binning Notebook` will find neutron count rates over time and output to `3-Output/ID-XXX`

## Contributors

[Ryan Oldford, B.CS](https://github.com/ROldford)

[Ben Luginbuhl](https://github.com/bluginbuhl)

[Alvin Hendricks](https://github.com/A5H-git)

Sergey Issinski
