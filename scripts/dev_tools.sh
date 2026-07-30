#!/bin/bash

# this is a simple dev tools script for linting and testing

set -e  # exit immediately if a command exits with a non-zero status

# function to run linters
run_linters() {
    echo "running linters..."
    flake8 src/  # check python style
    black --check src/  # check formatting
}

# function to run tests
run_tests() {
    echo "running tests..."
    pytest tests/  # running tests
}

# function to build docker image
build_docker_image() {
    echo "building docker image..."
    docker build -t rlhf-llm-optimization .  # build your image here
}

# function to run local environment
run_local() {
    echo "running local environment..."
    python -m src.main  # assuming main is where your entry point is
}

# check the command passed
case "$1" in
    lint)
        run_linters
        ;;
    test)
        run_tests
        ;;
    build)
        build_docker_image
        ;;
    run)
        run_local
        ;;
    *)
        echo "usage: $0 {lint|test|build|run}"  # show usage
        exit 1
        ;;
esac

# TODO: consider adding more commands or options in the future
echo "done"  # signal completion