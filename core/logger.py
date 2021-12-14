"""
Logging Module

Module Description
==================
A utility module for logging.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""


from termcolor import colored
from typing import NoReturn
import sys

QUIET = "--quiet" in sys.argv

_state = {"incomplete": False}


def panic(message: str) -> NoReturn:
    if message_is_pending_completion():
        complete_err()

    print(colored("[ERROR] "+message, "red"))
    quit()


def err(message: str) -> None:
    print(colored("[ERROR] ", "red") + message)


def info(message: str, incomplete: bool = False) -> None:
    """
    Print a colored log message to the console. If the "incomplete" flag is set,
    the message will end with "..." and no newline character will be printed. Use functions
    complete_ok, complete_done or complete_err to finish the message later.
    """
    if QUIET:
        return
    terminator = "..." if incomplete else "\n"

    print(colored("[INFO] ", "cyan") + message, end=terminator)

    if incomplete:
        _state["incomplete"] = True
        sys.stdout.flush()


def complete_ok(msg: str = "ok") -> None:
    print(colored(msg, "green"))
    _state["incomplete"] = False


def complete_done(msg: str = "done") -> None:
    print(colored(msg, "cyan"))
    _state["incomplete"] = False


def complete_err(msg: str = "failed") -> None:
    print(colored(msg, "red"))
    _state["incomplete"] = False


def message_is_pending_completion() -> bool:
    """
    Return true if we've just sent an "incomplete" message and haven't added an "ok" or "done"
    at the end.
    """
    return _state["incomplete"]


def warn(message: str) -> None:
    if QUIET:
        return
    print(colored("[WARN] ", "yellow") + message)
