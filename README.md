# Thunderbird PSD

This repository includes code for processing, analysing, and visualising data from the Thunderbird scintillation detector. The primary purpose is for neutron and gamma-ray differentiation using pulse-shape discrimination (PSD).

<details>
    <summary>Installation</summary>

Clone the repository into a new directory on your computer.
(The `--recursive` flag is needed to also clone the `sample_datasets` submodule.)
```
git clone --recursive git@github.com:berlinguette/thunderbird_psd.git
```

Create a virtual environment and activate it

```
cd thunderbird_psd
python3 -m venv .venv --prompt=tbird_psd
```

on Windows:

```
.\.venv\Scripts\activate
```

on Linux/MacOS:

```
source .venv/bin/activate
```

Install the packages in `requirements.txt`

```
pip install -r requirements.txt
```
</details>

<details>
    <summary>Sample Datasets Update</summary>

Get updates to the sample datasets

```
git submodule update
```
</details>