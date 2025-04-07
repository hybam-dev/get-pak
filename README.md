## GET-pak
**G**éosciences **E**nvironnement **T**oulouse - **P**rocessing and **a**nalysis Wor**k**bench

GETpak aims to provide tools for Sentinel-2, Sentinel-3, GeoTIFF, NetCDF and vector data manipulation and validation.

```plaintext
            _..._
          .'     '.      _
         /    .-""-\   _/ \ 
       .-|   /:.   |  |   | 
       |  \  |:.   /.-'-./ 
       | .-'-;:__.'    =/  ,ad8888ba,  88888888888 888888888888                          88
       .'=  *=|CNES _.='  d8"'    `"8b 88               88                               88
      /   _.  |    ;     d8'           88               88                               88
     ;-.-'|    \   |     88            88aaaaa          88        8b,dPPYba,  ,adPPYYba, 88   ,d8
    /   | \    _\  _\    88      88888 88"""""          88 aaaaaa 88P'    "8a ""     `Y8 88 ,a8"
    \__/'._;.  ==' ==\   Y8,        88 88               88 """""" 88       d8 ,adPPPPP88 8888[
    /|\  /|\ \    \   |   Y8a.    .a88 88               88        88b,   ,a8" 88,    ,88 88`"Yba,
   / | \/ | \/    /   /    `"Y88888P"  88888888888      88        88`YbbdP"'  `"8bbdP"Y8 88   `Y8a
  /  | || |  /-._/-._/                                            88
             \   `\  \                                            88
              `-._/._/
```
### Setup
⚠️ GDAL is a requirement for the installation, therefore, 
usage of a conda environment 
([Anaconda.org](https://www.anaconda.com/products/individual)) 
is strongly recommended. Unless you know what you are doing (-:

## Installation
Create a Conda environment (python versions above 3.9 were not tested but they should also be compatible):
```
conda create --name getpak python=3.9
```
Activate your conda env:
```
conda activate getpak
```
With your conda env activated, install GDAL before installing `getpak` to avoid dependecy errors:
```
conda install -c conda-forge gdal
```
Clone `getpak` repo to the desired location. First, install the dependencies using conda:
```
conda install --file requirements.txt
```
Finally, install the library:
```
pip install .
```

To ensure installation, you can check the vignettes made for the library. Don't forget to correct the
path to the images!
