#!/bin/bash

# this script will help with linting and testing the codebase

set -e # exit immediately if a command exits with a non-zero status

# function to run linting
run_lint() {
    echo "running linter..."
    flake8 src/ # checking for PEP 8 compliance
    echo "linting completed"
}

# function to run tests
run_tests() {
    echo "running tests..."
    pytest tests/ # running all tests in the tests directory
    echo "all tests passed"
}

# function to build docker image
build_docker() {
    echo "building docker image..."
    docker build -t rlhf-llm-optimization . # change the image name as needed
    echo "docker image built successfully"
}

# function to run the application locally
run_local() {
    echo "running the application locally..."
    python src/main.py # replace with the actual entry point
    echo "application is now running"
}

# main script logic
case "$1" in
    lint)
        run_lint
        ;;
    test)
        run_tests
        ;;
    build)
        build_docker
        ;;
    run)
        run_local
        ;;
    *)
        echo "usage: $0 {lint|test|build|run}" # show usage when called with no arguments
        exit 1
        ;;
esac

# TODO: add more functionalities as needed