
HOW THE DASHBOARD RUNS — STEP BY STEP
--------------------------------------
 
Step 1  — User opens the dashboard URL in any browser
 
Step 2  — Dashboard loads Kore2_battery_performance.csv
          (8,640 rows across 12 batteries)
 
Step 3  — Data is cleaned using linear interpolation
          (fills 5% missing sensor values)
 
Step 4  — Features are engineered per battery:
          drain rate, ride duration,
          shutdown SOC, voltage std deviation, SOH
 
Step 5  — KMeans clusters all 12 batteries
          into 2 groups: Faulty vs Healthy
 
Step 6  — Fleet Overview page renders:
          drain rate bar chart
          SOH scatter plot
          SOC trajectory lines
          red alert banners for 4 faulty batteries
 
Step 7  — User clicks Battery Drill-Down
          selects one battery from dropdown
          sees SOC, voltage, speed, current charts
 
Step 8  — User goes to Live Scorer
          enters sensor readings manually
          Random Forest classifies in under 1 second
          SMS sent to rider via Africa's Talking API
 
 
