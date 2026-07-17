#!/bin/bash

# this script is for running development tools like linting and testing

set -e  # exit immediately if a command exits with a non-zero status

# function to run linter
run_lint() {
    echo "running linter..."
    flake8 src/  # check the source directory for style issues
    echo "linting completed"
}

# function to run tests
run_tests() {
    echo "running tests..."
    pytest tests/  # execute the tests in the tests directory
    echo "tests completed"
}

# function to build docker image
build_docker() {
    echo "building docker image..."
    docker build -t rlhf-llm-optimization .  # build the docker image
    echo "docker image built"
}

# check command line arguments
case "$1" in
    lint)
        run_lint
        ;;
    test)
        run_tests
        ;;
    docker)
        build_docker
        ;;
    *)
        echo "usage: $0 {lint|test|docker}"  # provide usage info for the script
        exit 1
        ;;
esac
# TODO: consider adding more commands as needed