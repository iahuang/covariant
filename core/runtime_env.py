"""
Module Environment Verification

Module Description
==================
A module providing tools for verifying the module environment, including
checking that all required modules are installed.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

import fs
import command as cmd
import core.util as util
import re


def _read_reqfile(path: str="requirements.txt") -> list[str]:
    """
    Return the names of the modules listed in the requirements file
    """

    reqs = fs.read_file_as_lines(path, remove_blank_lines=True)

    # Filter any commented lines (lines that start with "#")
    reqs = [line for line in reqs if not line.lstrip().startswith("#")]

    # Return just the name of the module; strip any version information
    module_names: list[str] = []

    for module_info in reqs:
        name_match = re.match(r'\w+', module_info)

        if name_match is not None:
            module_names.append(name_match.group(0))

    return module_names


def _verify_modules() -> list[str]:
    """
    Verify that all modules listed in the requirements file are installed.
    Returns a list of modules that have not been installed
    """

    # Use the pip CLI to return a list of installed modules
    installed_mods = cmd.run_command(["pip", "list"]).stdout.split("\n")
    installed_mods = [
        util.unwrap(re.match(r"[\w-]+", mod)).group(0) # Match just the module name and not the version info
        for mod in installed_mods
        if re.match(r"[\w-]+\s+[\d\.]+", mod)          # Search output string for modules
    ]

    # Get a list of required modules
    required_mods = _read_reqfile()

    # Returns a list of modules that have not been installed
    return list(set(required_mods).difference(installed_mods))
