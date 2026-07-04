#!/bin/bash

# script for development tools like linting and testing
set -e  # exit immediately if a command exits with a non-zero status

# function to lint the Python files
lint() {
    echo "running linter..."
    flake8 src/  # check for style guide enforcement
    echo "linting complete"
}

# function to run tests
test() {
    echo "running tests..."
    pytest tests/  # execute tests in the tests directory
    echo "testing complete"
}

# function to build and run docker container
docker_run() {
    echo "building and running docker container..."
    docker build -t rlhf_llm_opt .  # build the image
    docker run --rm -it rlhf_llm_opt  # run the container
    echo "docker run complete"
}

# help message
help() {
    echo "dev_tools.sh - development tools for RLHF-LLM-Optimization"
    echo "Usage: ./dev_tools.sh [command]"
    echo "Commands:"
    echo "  lint        run the linter"
    echo "  test        run the tests"
    echo "  docker      build and run the docker container"
    echo "  help        show this help message"
}

# check for command arguments
if [ $# -eq 0 ]; then
    echo "no command provided, showing help"
    help
    exit 1
fi

# main switch case
case $1 in
    lint)
        lint
        ;;
    test)
        test
        ;;
    docker)
        docker_run
        ;;
    help)
        help
        ;;
    *)
        echo "unknown command: $1"
        help
        exit 1
        ;;
esac

# TODO: consider adding more dev tools in the future