# OOPS_JAVA_ASSIGNMENT

# Yashpreet Singh
# 23BDS072

## Introduction
This Java program performs multivariate linear regression to predict house prices based on house size and year built. Here's a detailed explanation of each part of the code:

## Class Variables and Constructor
The multivariate_regression1 class includes:

beta: Array holding the coefficients (slopes) for each predictor variable.
intercept: Intercept term (constant term) of the regression model.
r2: R-squared value, representing the proportion of variance explained by the model.
svar0: Variance of the residuals, a measure of error in predictions.
The constructor multivariate_regression1(double[][] X, double[] y) takes two inputs:

X: A 2D array where each row is an observation and each column is a predictor (house size and year).
y: Array of response variables (house prices).
Steps in the Constructor

## Input Validation: 
Ensures the number of rows in X matches the length of y.
Matrix Setup: Constructs the design matrix XtX and vector Xty used for regression calculations.
Matrix Multiplication: Builds XtX (X-transpose times X) and Xty (X-transpose times y).
Solve for Coefficients: Calls the solve method to find the values of beta and intercept.
Model Evaluation: Calculates R-squared (r2) to determine how well the model explains the data.
Matrix Solver (solve)
This function solves the system of linear equations to compute the coefficients of the regression model. It uses Gaussian elimination with partial pivoting:

## Pivoting:
Selects the row with the largest value to avoid numerical errors.
Row Operations: Updates rows to make the matrix upper-triangular.
Back Substitution: Computes the coefficients by solving from the bottom row up.
Prediction and Evaluation
predict: Given an array of predictor values, this method calculates the predicted response (house price).
intercept and coefficients: Getters for retrieving the model's intercept and slopes.
R2: Returns the R-squared value, indicating the goodness of fit.
Data Loading Methods
readPredictors: Reads the predictor data from a CSV file. It handles both commas and periods as decimal points, and warns if invalid data is encountered. It also prints a summary of the data.
readResponse: Reads the response data (house prices) from a separate CSV file, with similar error handling and a summary printout.
Main Method (main)
File Paths: Defines file paths for the predictor and response data files.
Data Loading: Reads the predictor variables and response variables from files.
Model Instantiation: Creates an instance of multivariate_regression1 using the loaded data.
Output Model Statistics: Displays the intercept, coefficients, and R-squared of the model.
Interactive Predictions: Allows the user to input house size and year built to predict house prices in a loop. Exits when negative values are entered.
Confidence Metrics: Checks how close input values are to the mean of training data, indicating the prediction's reliability.
This code is essentially a simple linear regression tool that can be adapted for predicting any response variable given multiple predictors, with detailed input validation, error handling, and debugging output.
