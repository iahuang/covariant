"""
Codebase Checker

Module Description
==================
This module contains no functionality related to the project; it is here only to check the codebase
for code quality issues, etc.

Copyright Information
===============================
This file is Copyright (c) 2021 Deon Chan, Ian Huang, Emily Wan, Angela Zhuo.
"""

import ast
from glob import glob
import os
from typing import Union
from termcolor import colored
import core.fs as fs

total_lines_of_code = 0


def check_for_function_decl(node: Union[ast.AST, ast.Module], path: str):
    """
    Use the AST module to check for functions lacking a return signature.
    """
    if isinstance(node, ast.FunctionDef):
        args = node.args
        return_type = node.returns
        function_identifier = colored(node.name, "cyan") + " in " + colored('"{}"'.format(path), "green")\
            + " (line {})".format(node.lineno)

        # print(return_type)
        # print(args.args)
        
        if return_type is None and node.name != "__init__":
            print("Function "+function_identifier+" lacks a return signature")

    if hasattr(node, "body"):
        for node in getattr(node, "body"):
            check_for_function_decl(node, path)

# This file should not be imported directly

for path in glob("*.py"):
    if os.path.abspath(path) == __file__:
        continue

    source_code = fs.read_file(path)
    total_lines_of_code += source_code.count("\n") + 1

    root = ast.parse(source_code)
    check_for_function_decl(root, path)

print(total_lines_of_code)
