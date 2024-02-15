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

Set up notebook cleaning filters:

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.required true
git add --renormalize .
```

## Usage

TODO

Open Jupyter Lab

```bash
jupyter lab
```

Find main notebook in `active_notebooks`
Use `Run All`
Get output in `3-Output/ID-XXX`

## Contributors

[Ryan Oldford, B.CS](https://github.com/ROldford)

[Ben Luginbuhl](https://github.com/bluginbuhl)

[Alvin Hendricks](https://github.com/A5H-git)

Sergey Issinski
