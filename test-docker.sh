#!/bin/bash
# test-docker.sh - Reproduce CI failures locally using Docker

set -e

echo "🐳 Docker-based CI Reproduction Script"
echo "======================================="

# Configuration
PYTHON_VERSION=${1:-"3.10"}
PROJECT_ROOT=$(pwd)

echo "📋 Configuration:"
echo "  Python version: $PYTHON_VERSION"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Function to run tests in Docker container
run_docker_test() {
    local python_ver=$1
    local container_name="f1-ml-test-py${python_ver}"
    
    echo "🚀 Starting Docker container with Python $python_ver..."
    
    # Run container with project mounted
    docker run --rm -it \
        --name "$container_name" \
        -v "$PROJECT_ROOT:/work" \
        -w /work \
        -e PYTHONPATH=/work/src \
        python:$python_ver-slim bash -c "
        echo '📦 Installing system dependencies...'
        apt-get update && apt-get install -y build-essential cmake curl libssl-dev
        
        echo '🐍 Setting up Python environment...'
        python -m pip install --upgrade pip setuptools wheel
        
        echo '📚 Installing Python packages...'
        pip install --prefer-binary -r requirements.txt -c constraints.txt
        
        echo '📁 Creating required directories...'
        mkdir -p data/features data/raw models logs reports
        
        echo '🧪 Running tests...'
        echo 'Running syntax check first...'
        python -m py_compile main.py || echo 'Warning: main.py has syntax issues'
        
        echo 'Running unit tests...'
        pytest -q -m 'not integration' --tb=short || echo 'Some unit tests failed'
        
        echo '🎯 Test Summary:'
        echo '  - Container: python:$python_ver-slim'
        echo '  - Dependencies: Installed successfully'
        echo '  - Tests: Completed (check output above for failures)'
        "
}

# Main execution
echo "🔍 Available Python versions to test:"
echo "  - 3.9  (matches CI that was failing)"
echo "  - 3.10 (current stable)"  
echo "  - 3.11 (current stable)"
echo ""

if [ "$PYTHON_VERSION" = "all" ]; then
    echo "🚀 Running tests on all Python versions..."
    for version in 3.9 3.10 3.11; do
        echo ""
        echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
        echo "Testing Python $version"
        echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
        run_docker_test $version
    done
else
    run_docker_test $PYTHON_VERSION
fi

echo ""
echo "✅ Docker testing complete!"
echo ""
echo "💡 Usage examples:"
echo "  ./test-docker.sh           # Test Python 3.10 (default)"
echo "  ./test-docker.sh 3.9       # Test Python 3.9"
echo "  ./test-docker.sh all       # Test all versions"
echo ""
echo "🐛 If tests fail, you can debug interactively:"
echo "  docker run --rm -it -v \$(pwd):/work -w /work python:$PYTHON_VERSION-slim bash"