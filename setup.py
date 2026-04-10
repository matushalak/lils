from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"

# Top-level modules in scripts/ (e.g., device.py, warnings_setup.py)
py_modules = [p.stem for p in SCRIPTS_DIR.glob("*.py") if p.is_file()]

setup(
    name="lils",
    version="0.1.0",
    package_dir={"": "scripts"},
    packages=find_namespace_packages(where="scripts"),
    py_modules=py_modules,
)
