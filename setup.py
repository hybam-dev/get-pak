import getpak
from setuptools import setup, find_packages

__version__ = getpak.__version__
__package__ = 'getpak'
short_description = 'Sentinel-2 and 3 raster and vector manipulation and validation tools.'

setup(
    name=__package__,
    version=__version__,
    url='https://github.com/daviguima/get-pak',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_data={
        '': ['*.json'],
        'getpak': ['getpak/data/*']
        },
    include_package_data=True,

    license='GPLv3',
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
