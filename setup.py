from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="testframework",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'pytest-xdist>=3.0.0',
            'codecov>=2.1.0',
            'numpy<2.0.0',  # Required for deepchecks compatibility
            'deep-eval>=0.14.0;python_version<"3.12"',  # Deep-eval doesn't support Python 3.12 yet
            'deepchecks>=0.19.1',  # For ML model validation and testing
        ],
    },
    python_requires='>=3.8',
)
