import re
from pathlib import Path

import click

from jobfile_generator.pbs_credentials import PbsCreds

pbs_creds = PbsCreds()  # type: ignore VSCode, .env will populate params


@click.command()
def generate_jobfile():
    walltime = get_formatted_walltime()

    cpus = click.prompt("How many CPUs do you want", default=8, prompt_suffix="?")
    memory = click.prompt(
        "How much memory do you want (in GB)", default=64, prompt_suffix="?"
    )
    notify = click.confirm(
        "Do you want conversion status notifications by email",
        default=True,
        prompt_suffix="?",
    )
    if notify:
        email = click.prompt("What email address should we use", prompt_suffix="?")
    else:
        email = None

    pbs_file_lines = _generate_file_lines(walltime, cpus, memory, notify, email)
    # This needs to be run from repo root to work properly
    jobfile_path = Path(__file__).parent / "thunderbird_psd.pbs"
    with open(jobfile_path, "w") as jobfile:
        jobfile.writelines(pbs_file_lines)

    click.echo(f"Jobfile created at {jobfile_path}")


def get_formatted_walltime() -> str:
    valid = False
    walltime = "02:00:00"
    while not valid:
        prompt_result = click.prompt(
            "How much processing time do you need (as '?h?m')",
            default="2h",
            prompt_suffix="?",
        )
        prompt_result = prompt_result.lower().replace(" ", "")

        try:
            wall_hours = _parse_walltime(prompt_result, r"(\d+)h", "Hours")
        except (ValueError, IndexError):
            continue

        try:
            wall_minutes = _parse_walltime(prompt_result, r"(\d+)m", "Minutes")
        except (ValueError, IndexError):
            continue

        if wall_hours == 0 and wall_minutes == 0:
            click.echo("Please enter a non-zero time in the correct format ('?h?m').")
            continue

        walltime = f"{wall_hours:02}:{wall_minutes:02}:00"
        valid = True
    return walltime


def _parse_walltime(walltime: str, regex_string: str, time_section: str):
    time_section = time_section.capitalize()

    match = re.search(regex_string, walltime)
    if match:
        try:
            walltime_section = int(match.group(1))
        except (ValueError, IndexError):
            click.echo(f"That value was invalid. {time_section} must be an integer")
            raise
    else:
        walltime_section = 0

    return walltime_section


def _generate_file_lines(
    walltime_str: str, cpus: int, memory: int, notify: bool, email: str | None = None
):
    if notify and email is not None:
        notify_line = "#PBS -m abe"
        email_line = f"#PBS -M {email}"
    else:
        notify_line = None
        email_line = None
    alloc_code = pbs_creds.alloc_code

    pbs_file_lines: list[str | None] = [
        "#!/bin/bash",
        "",
        f"#PBS -l walltime={walltime_str},select=1:ncpus={cpus}:mem={memory}gb",
        "#PBS -N thunderbird_psd",
        f"#PBS -A {alloc_code}",
        notify_line,
        email_line,
        "#PBS -W umask=007",
        f"#PBS -W group_list={alloc_code}-rw",
        "",
        "################################################################################",
        "",
        "# Change directory into the job dir",
        "cd $PBS_O_WORKDIR",
        "",
        "# Load software environment",
        "module load gcc",
        "module load apptainer",
        "",
        "# Set RANDFILE location to writeable dir",
        "export RANDFILE=$TMPDIR/.rnd",
        "",
        "# Generate a unique token (password) for Jupyter Notebooks",
        "export APPTAINERENV_JUPYTER_TOKEN=$(openssl rand -base64 15)",
        "",
        "# Find a unique port for Jupyter Notebooks to listen on",
        "readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("
        ", 0)); print(s.getsockname()[1]); s.close()')",
        "",
        "# Print connection details to file",
        "cat > connection_${PBS_JOBID}.txt <<END",
        "",
        "1. Create an SSH tunnel to Jupyter Notebooks from your local workstation using the following command:",
        "",
        "ssh -N -L 8888:${HOSTNAME}:${PORT} ${USER}@sockeye.arc.ubc.ca",
        "",
        "2. Point your web browser to http://localhost:8888",
        "",
        "3. Login to Jupyter Notebooks using the following token (password):",
        "",
        "${APPTAINERENV_JUPYTER_TOKEN}",
        "",
        "When done using Jupyter Notebooks, terminate the job by:",
        "",
        "1. Quit or Logout of Jupyter Notebooks",
        "2. Issue the following command on the login node (if you did Logout instead of Quit):",
        "",
        "qdel ${PBS_JOBID}",
        "",
        "END",
        "",
        "# Execute jupyter within the jupyter/datascience-notebook container",
        "apptainer exec \\",
        f"--home /scratch/{alloc_code}/jobs/thunderbird_psd \\",
        f"/arc/project/{alloc_code}/jupyter/jupyter_parquet.sif \\",
        "jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0 --notebook-dir=$PBS_O_WORKDIR,",
    ]
    return [f"{line}\n" for line in pbs_file_lines if line is not None]


if __name__ == "__main__":
    generate_jobfile()
