from setuptools import setup, find_packages

setup(
    name="testframework",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your project's dependencies here
    ],
)
