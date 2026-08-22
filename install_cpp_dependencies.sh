#!/usr/bin/env bash
# Installs the C++ build dependencies required by Makefile and setup.py.

set -euo pipefail

case "$(uname -s)" in
    Darwin)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Homebrew is required. Install it from https://brew.sh, then run this script again." >&2
            exit 1
        fi

        if ! xcode-select -p >/dev/null 2>&1; then
            echo "Installing the Xcode Command Line Tools..."
            xcode-select --install
            echo "Complete the installation, then run this script again." >&2
            exit 1
        fi

        brew install eigen libomp
        ;;
    Linux)
        if ! command -v apt-get >/dev/null 2>&1; then
            echo "Unsupported Linux package manager. Install a C++20 compiler, Eigen 3, and OpenMP manually." >&2
            exit 1
        fi

        if [ "$(id -u)" -eq 0 ]; then
            apt-get update
            apt-get install -y build-essential libeigen3-dev libomp-dev
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y build-essential libeigen3-dev libomp-dev
        else
            echo "Run this script as root, or install sudo, to install C++ dependencies." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported operating system. Install a C++20 compiler, Eigen 3, and OpenMP manually." >&2
        exit 1
        ;;
esac

echo "C++ dependencies installed. Run 'make all' to build and test the project."
