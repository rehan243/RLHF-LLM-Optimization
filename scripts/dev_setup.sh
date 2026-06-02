#!/bin/bash

# script to set up the development environment
# install dependencies if not already installed
function install_dependencies() {
    echo "installing dependencies..."
    pip install -r requirements.txt
}

# run linters
function run_linters() {
    echo "running linters..."
    flake8 src/
    pylint src/
}

# run tests
function run_tests() {
    echo "running tests..."
    pytest tests/
}

# check if dependencies are installed
if ! pip show -q -f requirements.txt; then
    install_dependencies
fi

# run linters and tests
run_linters
run_tests

echo "dev setup complete"