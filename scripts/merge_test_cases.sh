#!/bin/bash
PYTHON_BIN=${PYTHON_BIN:-$(which python)}
${PYTHON_BIN} -u merge_test_cases.py --method_name $1 --save_dir $2