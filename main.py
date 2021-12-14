"""
Main File

Module Description
==================
This module produces an interactive visualization of our model's predictions for COVID-19 cases and deaths.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

import sys
import os

SKIP_CHECKS = "--skip-check" in sys.argv or os.path.exists("_data/env_ok.txt")


def _raise_error_and_exit(msg: str) -> None:
    """Log an error to the console in red text and exit"""

    print("\033[31m"+msg+"\033[0m")
    print("\033[31mIf you are sure that the runtime environment is properly configured, \
you can run this program again with the command line argument --skip-check.\033[0m")

    quit()


def _env_check() -> None:
    """Verify that the runtime environment is properly configured"""

    print("Since this is the first time running this program, we will run preliminary \
checks to ensure the runtime environment is properly configured.")

    # CHECK PYTHON VERSION

    if sys.version_info[0] < 3:
        _raise_error_and_exit(
            "Error: This program must be run using Python 3.")
        quit()

    if sys.version_info[0] == 3 and sys.version_info[1] < 9:
        _raise_error_and_exit(
            "Error: This program requires Python 3.9 or later.")
        quit()

    if sys.version_info[1] == 10:
        print("WARNING: As of writing, Tensorflow does not support Python 3.10. Use with caution!")

    # VERIFY PIP INSTALLED

    import core.command as cmd

    if cmd.run_command(["pip", "--version"]).stderr:
        _raise_error_and_exit("Error: pip is not installed.")

    # VERIFY RUNTIME ENVIRONMENT

    import core.runtime_env as runtime_env

    not_installed = runtime_env._verify_modules()
    if not_installed:
        err = "Error: The following required packages are not installed:\n"

        for mod in not_installed:
            err += " - "+mod+"\n"

        _raise_error_and_exit(err)


def bootstrap() -> None:
    """
    A function that wraps the main() function and performs preflight checks.
    """
    # If necessary, perform preliminary checks before doing anything else

    if not SKIP_CHECKS:
        _env_check()

    # Mark that the preliminary checks passed
    import core.fs as fs
    fs.write_file("_data/env_ok.txt", "")

    # Begin program
    import core.logger as logger
    import traceback
    from termcolor import colored

    try:
        main()
    except KeyboardInterrupt:
        logger.info("User halted program, exiting...")
    except Exception as e:
        if logger.message_is_pending_completion():
            logger.complete_err("error")
        logger.err("An unhandled exception ocurred:\n")
        print(colored(traceback.format_exc(), "red"))


def main() -> None:
    import core.logger as logger
    from termcolor import colored

    # If necessary, download the datasets
    import core.dataset_downloader as ds

    if not os.path.exists("_data/dataset.json"):
        ds.build_dataset()

    # Read dataset file
    logger.info("Loading dataset", incomplete=True)
    import core.county as county
    dataset = county.CountyDataset()
    dataset.load_dataset("_data/dataset.json")
    logger.complete_done()

    logger.info("Loading Tensorflow (if CUDA is not properly configured, this may throw warnings. \
You may safely ignore these)...")
    import core.ml as ml
    logger.info("Tensorflow loaded successfully.")
    logger.info("Initializing model...")
    model = ml.COVIDGraphModel(dataset)
    logger.info("Model initialized successfully.")

    if not os.path.exists("_data/model") or "--re-train" in sys.argv:
        print(colored("Model needs to be trained (this may take a little while)", "cyan"))
        model.train(0.95, epochs=1)
        model.save()
    else:
        print(colored(
            "Model has already been trained (use ", "cyan")
            + "--re-train" + colored(" to re-train model).", "cyan"))
        model.load()
    
    import core.data_vis as data_vis
    logger.info("Opening data visualizer... (check your task bar if it does not focus \
automatically.")

    vis = data_vis.ModelVisualizer(model)
    vis.run()


if __name__ == "__main__":
    bootstrap()
