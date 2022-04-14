# Thunderbird PSD

This repository includes code for processing, analysing, and visualising data from the Thunderbird scintillation detector. The primary purpose is for neutron and gamma-ray differentiation.

<details>
    <summary>Installation</summary>

    Clone the repository into a new directory on your computer

    ```
    git clone git@github.com:berlinguette/thunderbird_psd.git
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