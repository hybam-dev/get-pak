import os
from setuptools import setup, find_packages

__package__ = 'getpak'
__version__ = '0.1.5'

short_description = 'Raster and vector manipulation toolbox for reproducible water quality research.'

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name=__package__,
    version=__version__,
    url="https://github.com/SNO-HYBAM/get-pak",
    packages=find_packages(include=["getpak", "getpak.*"]),
    py_modules=['main'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        '': ['*.json', '*.ini'],
        'getpak': ['getpak/data/*']
        },
    include_package_data=True,

    license='MIT',
    author='David Guimaraes',
    author_email='dvdgmf@gmail.com',
    description=short_description,
    entry_points={
        'console_scripts': ['getpak=main:main'],
    },
    install_requires=[
        'scikit_learn',
        'matplotlib',
        'numpy',
        'pandas',
        'rasterstats'
        ]
    )
