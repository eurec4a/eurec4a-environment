#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer

INSTALL_REQUIRES = ["xarray", "netcdf4", "typhon", "tqdm"]

setup(
    name="eurec4a-environment",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="tools for characterising the atmospheric environment during EUREC4A",
    url="https://github.com/eurec4a/eurec4a-environment",
    maintainer="Leif Denby",
    maintainer_email="l.c.denby@leeds.ac.uk",
    py_modules=["eurec4a_environment"],
    packages=find_packages(),
    package_data={"": ["*.csv", "*.yml", "*.html"]},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
)
