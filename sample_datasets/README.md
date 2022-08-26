# Sample Datasets

This directory contains sample "neutron experiment" datasets. Each experiment
consists of one or more "buffers", each of which consists of between 1 and
10,000 "signals."

The raw data for a given experiment is saved *via* the PicoScope6 software
package from Pico Technology. Each buffer is saved as a `.psdata` file (a
proprietary binary format) in the `raw_data/psdata/` directory for a given
experiment. Additionally, the time taken to collect each buffer and save it to
disk is recorded in `raw_data/exp_times.csv`. Any relevant metadata for an
experiment is recorded in `exp_info.txt` (TODO: define a standard format for
essential experimental metadata).

Once an experiment is finished, *i.e.*, all of the buffers have been saved in
`.psdata` format, they are each converted into 
[Apache Parquet](https://parquet.apache.org/) format (`.parquet`) and stored in
the `raw_data/parquet/` directory using the `neutron_data_converter.py` script.

A given parquet file stores all of the signals in a buffer as rows in a
table-like format. Each signal is indexed and labelled according to the buffer
it is from and its number (out of the total signals in the buffer). For example,
the 1,234th (/10,000) signal from the **first buffer** of the `20220824`
experiment would be labelled as `b0001s01234`. The columns of the parquet
correspond to the amplitude of the signals at each time index.

> **Note:** because parquet format does not accept integers as column names,
> the column values in each parquet file are stored as strings. When loading the
> parquet files into a dataframe, it can sometimes be helpful to re-cast the
> column names as integers:
> ```
> df = pandas.read_parquet('/path/to/parquet_file.parquet')
> df.columns = df.columns.astype(int)
> ```
> Each signal is stored as a *row* rather than a *column* because it is ~10x
> faster to load the full dataset into a `DataFrame`.

## Experiment

### Buffer

#### Signal