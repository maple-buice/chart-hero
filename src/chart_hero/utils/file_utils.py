import os
from pathlib import Path


def get_first_match(dir: str, name_parts: list[str]) -> str:
    found_file = False
    for file in sorted(os.listdir(dir)):
        for name_part in name_parts:
            if name_part in file:
                found_file = True
            else:
                found_file = False
                break
        if found_file:
            return os.path.join(dir, file)

    raise Exception(f"No file matching '{name_part}' found in '{dir}'")


def get_dir(parent_dir: str, dir_name: str, create_if_not_found: bool) -> str:
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"parent_dir '{parent_dir}' does not exist")

    dir = os.path.join(parent_dir, dir_name)

    if not os.path.exists(dir):
        if create_if_not_found:
            print(f"Creating directory '{dir}'")
            os.makedirs(dir)
        else:
            raise FileNotFoundError(f"Directory '{dir}' does not exist")
    elif not os.path.isdir(dir):
        raise FileNotFoundError(f"Path '{dir}' is not a directory")

    return dir


def get_training_data_dir() -> str:
    dir_name = "training_data"

    try:
        return get_dir(os.getcwd(), dir_name, False)
    except FileNotFoundError:
        try:
            return get_dir(Path(os.getcwd()).parent, dir_name, False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Unable to locate directory '{dir_name}' in "
                + f"working directory: '{os.getcwd()}' or its parent"
            )


def get_model_training_dir() -> str:
    dir_name = "model_training"

    try:
        return get_dir(os.getcwd(), dir_name, False)
    except FileNotFoundError:
        try:
            return get_dir(Path(os.getcwd()).parent, dir_name, False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Unable to locate directory '{dir_name}' in "
                + f"working directory: '{os.getcwd()}' or its parent.\n"
                + "Please initiate training from the project root directory."
            )


def get_dataset_dir() -> str:
    training_data_dir = get_training_data_dir()
    data_set_dir_name = "e-gmd-v1.0.0"
    try:
        get_dir(training_data_dir, data_set_dir_name, False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Unable to locate trainind data directory '{data_set_dir_name}' in '{training_data_dir}'.\n"
            + "Model cannot be trained without data.\n"
            + "Please download the training data from https://magenta.tensorflow.org/datasets/e-gmd "
            + f"and extract it as '{data_set_dir_name}' into '{training_data_dir}'"
        )


def get_audio_set_dir() -> str:
    return get_dir(get_training_data_dir(), "audio_set", True)


def get_audio_set_files() -> list[str]:
    audio_set_dir = get_audio_set_dir()
    audio_set_files = []

    for file in os.listdir(audio_set_dir):
        if file.endswith(".pkl"):
            audio_set_files.append(os.path.join(audio_set_dir, file))

    return audio_set_files


def get_labeled_audio_set_dir() -> str:
    return get_dir(get_training_data_dir(), "labeled_audio_set", True)


def get_process_later_dir() -> str:
    return get_dir(get_training_data_dir(), "process_later", True)


def get_model_backup_dir() -> str:
    return get_dir(get_training_data_dir(), "model_backup", True)


def get_model_dir() -> str:
    return get_dir(get_model_training_dir(), "model", True)


def get_model_file() -> str:
    return os.path.join(get_model_dir(), "model.keras")
