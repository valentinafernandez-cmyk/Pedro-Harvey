## Environment Setup

To guarantee reproducibility (Python >= 3.8), this project uses **`uv`** for virtual environment and dependency management.

### 1. Prerequisites
Install `uv` if you haven't already:
* **Linux/macOS:** curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
* **Windows:** irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex

### 2. Configuration
Create your local environment file from the template and add your CoinGecko API key:
```bash
cp .env.example .env
```
### 3. Initialization & Usage

**Synchronize the environment:** (Creates .venv and installs exact dependencies from uv.lock)
```bash
uv sync
````