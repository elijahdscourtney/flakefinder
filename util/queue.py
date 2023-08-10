import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import re
from util.logger import logger


def load_queue(input_file_path: str) -> list[tuple[str, str]]:
    """
    Loads the queue file at the specified file path.
    :param input_file_path: The path to the queue file.
    :return: The parsed list of input and output directories, given in pairs of `[input_dir, output_dir]`
    """
    print(input_file_path)
    with open(str(input_file_path)) as file1:
        inputs = file1.readlines()
    print(inputs)
    config = [re.match(r"InputDir: (.+), OutputDir: (.+)", line).group(1, 2) for line in inputs]

    logger.info(f"Loaded config: {config}")
    return config
