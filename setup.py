from pathlib import Path
from setuptools import setup, find_packages


with open("./readme.md", "r") as ff:
    readme_text = ff.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Parse version
init = Path(__file__).parent / "mip_cvxpy" / "__init__.py"
version = None
for line in init.read_text().split("\n"):
    if line.startswith("__version__"):
        version = line.split("=")[-1].strip('"')
        break
if version is None:
    raise ValueError("No version found")

setup(
    name="mip-cvxpy",
    version=version,
    description="Solve MILP CVXPY problems using python-mip",
    python_requires=">=3.6",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    author="Michael Jurasovic",
    url="https://github.com/jurasofish/mip_cvxpy",
    license="MIT License",
    packages=find_packages(),
    # package_data={'sphinx_toggleprompt': ['_static/toggleprompt.js_t']},
    classifiers=["License :: OSI Approved :: MIT License"],
    install_requires=requirements,
)
