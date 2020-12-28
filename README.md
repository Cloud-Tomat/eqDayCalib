# eqDayCalib

**What?**

This script aims to perform equatorial mount periodic error chaterization during day time.

**Why?**

- Avoid wasting good observation night to perform such task
- To not wait to have a clear sky to check the result of mechanical adjustments

**How?**

The principle consit in shooting picture at regular interval with long focal lens/telescope while Right Ascencion motor tracking is running. 
Longer the focal will be, higher will be the measurment resolution.

The script does the following:
1. It calculates the displacement between the pictures,
2. It deduces from this displacement : Right Ascencsion Speed between pictures
3. It perform integration of these displacement to calculate 
