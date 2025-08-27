#!/bin/bash

# CorePulse MLX Demo Runner
# Run various demos to showcase CorePulse capabilities

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "   CorePulse MLX Demo Suite"
echo "================================"
echo ""

# Check if corpus-mlx is installed
if ! python -c "import corpus_mlx" 2>/dev/null; then
    echo "Installing corpus-mlx..."
    cd "$PROJECT_ROOT"
    pip install -e . --quiet
    echo "✓ Installation complete"
fi

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/demo_outputs"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Run demos based on argument
case "${1:-all}" in
    basic)
        echo "Running basic injection demo..."
        python "$PROJECT_ROOT/examples/demo_basic.py"
        ;;
    masked)
        echo "Running regional/masked control demo..."
        python "$PROJECT_ROOT/examples/demo_masked.py"
        ;;
    weighted)
        echo "Running prompt weighting demo..."
        python "$PROJECT_ROOT/examples/demo_weighted.py"
        ;;
    sdxl)
        echo "Running SDXL demo..."
        python "$PROJECT_ROOT/examples/demo_sdxl.py"
        ;;
    all)
        echo "Running all demos..."
        echo ""
        echo "1. Basic injection..."
        python "$PROJECT_ROOT/examples/demo_basic.py"
        echo ""
        echo "2. Regional control..."
        python "$PROJECT_ROOT/examples/demo_masked.py"
        echo ""
        echo "3. Prompt weighting..."
        python "$PROJECT_ROOT/examples/demo_weighted.py"
        echo ""
        echo "================================"
        echo "All demos complete!"
        echo "Results saved to: $OUTPUT_DIR"
        ;;
    *)
        echo "Usage: $0 [basic|masked|weighted|sdxl|all]"
        echo ""
        echo "Options:"
        echo "  basic    - Run basic prompt injection demo"
        echo "  masked   - Run regional/masked control demo"
        echo "  weighted - Run prompt weighting demo"
        echo "  sdxl     - Run SDXL-specific demo"
        echo "  all      - Run all demos (default)"
        exit 1
        ;;
esac

echo ""
echo "Demo outputs saved to: $OUTPUT_DIR"
echo "Done!"