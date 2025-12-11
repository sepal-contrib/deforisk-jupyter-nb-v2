#!/bin/bash

# Build script for Deforisk documentation

echo "Building Deforisk Analysis Framework Documentation"
echo "=================================================="

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Install/update dependencies
echo "Installing documentation dependencies..."
pip install -q -r requirements.txt

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Build HTML documentation
echo "Building HTML documentation..."
make html

# Check for build success
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Documentation built successfully!"
    echo ""
    echo "View the documentation:"
    echo "  Open: build/html/index.html"
    echo "  Or run: python -m http.server --directory build/html 8000"
    echo "  Then visit: http://localhost:8000"
else
    echo ""
    echo "✗ Documentation build failed!"
    exit 1
fi
