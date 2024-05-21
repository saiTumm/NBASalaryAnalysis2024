# NBASalaryAnalysis2024

It is currently a work in progress.

**Completed:**

1. Data Collection:
    1. Collected data from [BasketbalLR  ](https://www.basketball-reference.com/)
    2. Data included Player Bio Data, Teams, Contract Values, and Performance Metrics
    
2. Data Cleaning:
    1. Merged all data sets to create a comprehensive data set
        1. See CombinedNBAStatsSalaries.xlsx
    2. Removed empty NaN values and inconsistencies
  
4. Preliminary Analysis:
    1. Created calculated fields in Tableau to better find a correlation between statistics and salary
    2. Visualized this correlation in various Tableau Sheets
    3. Created a dashboard for future display
  
5. Statistical Analaysis:
    1. Trained 3 different machine learning models (Random Forest Regressor, Linear Regression, and KN Neighbors Regressor)
    2. Averaged them to get an aggregated predicted salary value
    3. Conducted a poll and garnered 50+ responses to get human sentiment on which salary seems correct
    4. Compared statistical metrics among the 4 models (mse, rmse, mae, r2)
    5. Visualized graphs of error, and variation between the actual and predicted salary measurement
  
**To be completed soon:**
1. Upload of all Tableau Sheets and Dashboards
2. Comprehensive detail added to Jupyter Notebook
3. Analysis of each Tableau Sheet and Dashboard
4. Case Study report on the conclusions drawn
5. A chatbot for users to enter stats and be given a predicted salary

**Upcoming Goals**
1. Website using React to display the analysis and report
2. Webpage that acts a salary calculator
