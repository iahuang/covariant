"""
Utility Module

Module Description
==================
General utility module.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

from typing import Type, TypeVar, Union
import sys

def with_unix_endl(string: str) -> str:
    """Return [string] with UNIX-style line endings"""

    return string.replace("\r\n", "\n").replace("\r", "")

T = TypeVar("T")

def unwrap(value: Union[T, None]) -> T:
    """
    Unrwap an optional value. Raises a ValueError if the value is None.
    """

    if value is None:
        raise ValueError("Tried to unwrap a None value")
    return value

def truncate(string: str, length: int) -> str:
    """
    If the provided string is longer than [length], the string is truncated and "..." is added.
    """

    if len(string) > length:
        return string[:length-3]+"..."
    
    return string

def user_warn(message: str) -> bool:
    """
    Present the user with a y/n dialogue message IF the -y CLI flag was not passed.
    Return whether the user chose yes or not.
    """
    from termcolor import colored

    if "-y" in sys.argv: return True

    print(colored(message, "yellow"))
    response = input(colored("Ok? y/n: ", "cyan")).strip().lower()

    if response == "y" or response == "yes":
        return True
    
    return False
