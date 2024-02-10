# Thunderbird Data Analysis 

This repository includes code for processing, analysing, and visualising data from the Thunderbird reactor. The primary purpose is to determine neutron count rates from EJ309 data, and process reactor data for comparison with count rates.

## Installation

Clone this repository: 

```
git clone git@github.com:berlinguette/thunderbird_psd.git
```

Create a virtual environment:

```
cd thunderbird_psd
python3 -m venv .venv --prompt=tbird_psd
```

Activate your virtual environment:
- On Windows:

```
.\.venv\Scripts\activate
```

- On Linux/MacOS:

```
source .venv/bin/activate
```

Install required packages:


```
pip install -r requirements.txt
```

Set up notebook cleaning filters:

```
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.required true
git add --renormalize .
```

## Usage

TODO

Open Jupyter Notebook
Find main notebook in `active_notebooks`
Use `Run All`
Get output in `3-Output/ID-XXX`

## Contributors

[Ryan Oldford, B.CS](https://github.com/ROldford)

[Ben Luginbuhl](https://github.com/bluginbuhl)

[Alvin Hendricks](https://github.com/A5H-git)

Sergey Issinski
