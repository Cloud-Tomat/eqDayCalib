# eqDayCalib

**What?**

This script aims to perform equatorial mount periodic error chaterization during day time.
It generates temporal and spectral result csv files and offer basic plots:



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

Install a vie
