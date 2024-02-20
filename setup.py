#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='mimicplay',
    packages=[
        package for package in find_packages()
        if package.startswith("mimicplay")
    ],
    version='1.0',
    description='',
    author='Chen Wang',
    author_email='chenwj@stanford.edu',
    url='https://mimic-play.github.io',
    include_package_data=True,
    python_requires='>=3',
)
