## Environment Setup

To guarantee reproducibility (Python $\ge 3.8$), this project uses **`uv`** for virtual environment and dependency management.

### Prerequisites
Install `uv` if you don't have it yet:
* **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `irm https://astral.sh/uv/install.ps1 | iex`

### Initialization & Usage

**Synchronize the environment:** This creates the local `.venv` and installs the dependencies listed in `uv.lock`.
   ```bash
   uv sync
   ```