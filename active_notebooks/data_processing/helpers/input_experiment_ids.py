from data_processing.paths import get_parq_root

def input_experiment_ids() -> list[str]:
    """Prompts the user for an experiment ID number.
    Multiple ID numbers can be entered.
    Each ID is formatted as "ID-{number}".
    Each final ID is checked to see if valid data can be found for that experiment.
    If any cannot be found, the user must try again

    :return: All experiment IDs
    :rtype: list[str]
    """
    done = False
    experiment_ids = []
    while not done:
        while (
            id_input := input(
                "Enter the ID number (just the number!), or Enter to finish: "
            )
        ) != '':
            experiment_ids.append(id_input)
        experiment_ids = [f"ID-{exp_id}" for exp_id in experiment_ids]

        ids_valid = []
        for exp_id in experiment_ids:
            id_valid = get_parq_root(exp_id).is_dir()
            ids_valid.append(id_valid)
            if not id_valid:
                print(f"Experiment {exp_id} cannot be found")

        done = all(ids_valid)
        if not done:
            print("Invalid experiment IDs, please re-enter")
            experiment_ids = []
        else:
            print("All experiment IDs are valid")
    return experiment_ids
