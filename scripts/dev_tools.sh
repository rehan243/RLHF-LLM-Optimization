#!/bin/bash

# check for required tools
if ! command -v flake8 &> /dev/null
then
    echo "flake8 could not be found, install it to continue"
    exit
fi

if ! command -v pytest &> /dev/null
then
    echo "pytest could not be found, install it to continue"
    exit
fi

# run flake8 for linting
echo "running flake8 for linting"
flake8 src/

# if linting fails, exit early
if [ $? -ne 0 ]; then
    echo "linting failed, please fix issues"
    exit 1
fi

echo "linting passed"

# run tests
echo "running tests with pytest"
pytest tests/

# if tests fail, notify user
if [ $? -ne 0 ]; then
    echo "tests failed, check the output above"
    exit 1
fi

echo "all tests passed"

# TODO: add docker build and run commands if needed in the future
echo "you can now continue with your development"