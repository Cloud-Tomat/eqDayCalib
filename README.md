# eqDayCalib

**Quick Start**



**What?**

This script aims to perform equatorial mount periodic error chaterization during day time.
It generates temporal and spectral result csv files and offer basic plots:

![Period Error Plot](https://github.com/Cloud-Tomat/eqDayCalib/blob/main/doc/Periodic_error.png)

![Period Error DFT](https://github.com/Cloud-Tomat/eqDayCalib/blob/main/doc/Dft.png)

**Why?**

- Avoid wasting good observation night to perform such task
- To not wait to have a clear sky to check the result of mechanical adjustments

**How?**

The principle consit in shooting picture at regular interval with long focal lens/telescope while Right Ascencion motor tracking is running. 
Longer the focal will be, higher will be the measurment resolution.

The script does the following:
- Calculate the displacement between the pictures,
- Deduce from this displacement Right Ascencsion Speed between pictures
- Neutralize tracking speed to only keep tracking speed error
- Perform integration of these trackings speed error  to calculate Position Error
- Calculate Discrete Fourrier Transform of Position Error
- Save all that to CSV and Graph it

**Procedure**

Prequisite :
- Equatorial mount with Right ascencion motorized !
- Digital Camera able to save picture with EXIF data
- Lens or telescope with long focal (200 to 2000 mm depending on pixel size of camera sensor)

Step 1: Take the pictures
- Install the Digital Camera and the lens telescope.
- Declination must be set to **90° or -90°** so the movement on the picture appear as a translation
- Pay special attention to have a good focus on all the range covered during the process
- Picture format JPEG is fine, pictures must contain **EXIF data**
- Take a picture each 5 to 10s longer focal requires shorter intervall
- Take picture to cover 3 to 4 turns of worm (35 to 45 mn)

Step 2 : Download and unzip the script file
The python script and its dependencies has been packaged with pyinstaller 
- Download the script here :
- Unzip it to some place you will remember 

Step 2 : preparation of directories
- Create 2 directories:
      - A first directory that contain the pictures and **only** the pictures (any other files will cause the script to stop)
      - A second working directory where the script will place the results file



