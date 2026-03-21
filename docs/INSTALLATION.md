# Installation Guide

## System Requirements

- **Operating System**: Ubuntu 22.04 (recommended)
- **Python Version**: 3.13
- **Conda Environment**: `poolenv`

## Step-by-Step Installation

### 1. Create Conda Environment

```bash
# Create conda environment with Python 3.13
conda create -n poolenv python=3.13
conda activate poolenv
```

### 2. Install PoolTool (Billiards Physics Engine)

```bash
# Clone the SJTU-RL2 pooltool repository
git clone https://github.com/SJTU-RL2/pooltool.git
cd pooltool

# Install poetry 2.2.1
pip install "poetry==2.2.1"

# Install pooltool from source with development dependencies
poetry install --with=dev,docs

# Return to project root
cd ..
```

### 3. Install Project Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install additional dependencies for baseline agents
pip install bayesian-optimization numpy

# Install the project in development mode (optional)
pip install -e .
```

### 4. Verify Installation

```bash
# Test if pooltool is installed correctly
python -c "import pooltool as pt; print('PoolTool installed successfully')"

# Test if CueZero imports work
python -c "import cuezero; print('CueZero imported successfully')"
```

## Environment Variables

No special environment variables required, but ensure your conda environment is activated:

```bash
conda activate poolenv
```

## Troubleshooting

### Poetry Installation Issues

If poetry installation fails, try:

```bash
pip install --upgrade pip
pip install "poetry==2.2.1" --no-cache-dir
```

### PoolTool Compilation Errors

Ensure you have build essentials installed:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake
```

### Python Version Mismatch

Make sure you're using Python 3.13:

```bash
python --version
# Should show Python 3.13.x
```

## Quick Start After Installation

See [Quick Start](../README.md#-quick-start) in the main README to run your first game.
