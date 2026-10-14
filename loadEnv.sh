# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
# Linux / macOS:
source .venv/bin/activate

# Windows CMD:
# .venv\Scripts\activate.bat

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 3. Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# 4. Install all pinned dependencies
python -m pip install -r requirements.txt