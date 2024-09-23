from data_processing.paths import get_parq_root
from helpers.get_input_with_default import get_input_with_default

def input_experiment_ids() -> list[str]:
    done = False
    experiment_ids: list[str] = []
    while not done:
        id_inputs = []
        while (
            id_input := input(
                "Enter the ID number (just the number!), or Enter to finish: "
            )
        ) != '':
            id_inputs.append(id_input)
        possible_ids: list[tuple[str, str]] = [(f"TB-{exp_id}", f"ID-{exp_id}") for exp_id in id_inputs]

        for exp_ids in possible_ids:
            tb_id, old_id = exp_ids
            tb_id_valid = get_parq_root(tb_id).is_dir()
            old_id_valid = get_parq_root(old_id).is_dir()
            if tb_id_valid:
                if old_id_valid:
                    which_id_input = get_input_with_default(f"Do you want to use {tb_id} instead of {old_id}? [Y/n]", "y", str)
                    if which_id_input.lower() == "y":
                        experiment_ids.append(tb_id)
                    else:
                        experiment_ids.append(old_id)
                experiment_ids.append(tb_id)
            elif old_id_valid:
                experiment_ids.append(old_id)
            else:
                experiment_ids.append("")
                print(f"Experiment {tb_id}/{old_id} cannot be found")

        ids_valid = [exp_id != "" for exp_id in experiment_ids]
        done = all(ids_valid)
        if not done:
            print("Invalid experiment IDs, please re-enter")
            experiment_ids = []
        else:
            print("All experiment IDs are valid")
    return experiment_ids
