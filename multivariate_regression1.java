import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class multivariate_regression1 
{
    private final double[] beta;
    private final double intercept;
    private final double r2;
    private final double svar0;

    public multivariate_regression1(double[][] X, double[] y) 
    {
        if (X.length != y.length)
         {
            throw new IllegalArgumentException("Row count of X and length of y must be equal.");
        }

        int n = y.length;
        int p = X[0].length;

        double[][] XtX = new double[p + 1][p + 1];
        double[] Xty = new double[p + 1];

        for (int i = 0; i < n; i++) {
            double[] xi = new double[p + 1];
            xi[0] = 1.0;
            System.arraycopy(X[i], 0, xi, 1, p);

            for (int j = 0; j < p + 1; j++) {
                Xty[j] += xi[j] * y[i];
                for (int k = 0; k < p + 1; k++)
                {
                    XtX[j][k] += xi[j] * xi[k];
                }
            }
        }

        double[] coefficients = solve(XtX, Xty);
        intercept = coefficients[0];
        beta = Arrays.copyOfRange(coefficients, 1, coefficients.length);

        double sumY = Arrays.stream(y).sum();
        double yBar = sumY / n;
        double totalSumSquares = 0.0, residualSumSquares = 0.0;

        for (int i = 0; i < n; i++) 
        {
            double fitted = predict(X[i]);
            totalSumSquares += (y[i] - yBar) * (y[i] - yBar);
            residualSumSquares += (y[i] - fitted) * (y[i] - fitted);
        }

        r2 = 1.0 - residualSumSquares / totalSumSquares;
        svar0 = residualSumSquares / (n - p - 1);
    }

    private double[] solve(double[][] A, double[] b) 
    {
        int n = b.length;
        for (int i = 0; i < n; i++)
         {
            int max = i;
            for (int j = i + 1; j < n; j++) 
            {
                if (Math.abs(A[j][i]) > Math.abs(A[max][i])) max = j;
            }
            double[] temp = A[i];
            A[i] = A[max];
            A[max] = temp;
            double t = b[i];
            b[i] = b[max];
            b[max] = t;

            for (int j = i + 1; j < n; j++) 
            {
                double factor = A[j][i] / A[i][i];
                b[j] -= factor * b[i];
                for (int k = i; k < n; k++) 
                {
                    A[j][k] -= factor * A[i][k];
                }
            }
        }

        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) 
            {
                sum += A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }
        return x;
    }

    public double intercept() 
    {
        return intercept;
    }

    public double[] coefficients() 
    {
        return beta;
    }

    public double R2() 
    {
        return r2;
    }

    public double predict(double[] x) 
    {
        double prediction = intercept;
        for (int j = 0; j < beta.length; j++) 
        {
            prediction += beta[j] * x[j];
        }
        return prediction;
    }

    /**
     * Reads predictor variables from a CSV file.
     */
    public static double[][] readPredictors(String filePath) throws IOException
    {
        List<double[]> predictorsList = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        
        if (lines.isEmpty()) 
        {
            throw new IOException("Empty file: " + filePath);
        }
        
        // Print the header for debugging
        System.out.println("Header row: " + lines.get(0));
        
        // Get the number of columns from the header
        int numColumns = lines.get(0).split(",").length;
        
        // Skip the header row by starting from index 1
        for (int i = 1; i < lines.size(); i++) 
        {
            String line = lines.get(i).trim();
            if (!line.isEmpty()) 
            {
                String[] values = line.split(",");
                double[] predictors = new double[numColumns];
                try 
                {
                    for (int j = 0; j < numColumns; j++)
                    {
                        // Handle both comma and period as decimal separators
                        String normalizedValue = values[j].trim().replace(",", ".");
                        predictors[j] = Double.parseDouble(normalizedValue);
                    }
                    predictorsList.add(predictors);
                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) 
                {
                    System.err.println("Warning: Invalid data at line " + (i + 1) + ": '" + line + "'");
                }
            }
        }
        
        if (predictorsList.isEmpty()) 
        {
            throw new IOException("No valid data found in file: " + filePath);
        }
        
        // Print summary of read data
        System.out.println("\nSuccessfully read " + predictorsList.size() + " rows of predictor variables");
        System.out.println("First few rows:");
        for (int i = 0; i < Math.min(3, predictorsList.size()); i++)
        {
            System.out.print("Row " + (i + 1) + ": ");
            System.out.println(Arrays.toString(predictorsList.get(i)));
        }
        
        return predictorsList.toArray(new double[0][0]);
    }

    public static double[] readResponse(String filePath) throws IOException 
    {
        List<Double> responseList = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        
        if (lines.isEmpty()) 
        {
            throw new IOException("Empty file: " + filePath);
        }
        
        // Print the first few lines for debugging
        System.out.println("First few lines of response file:");
        for (int i = 0; i < Math.min(5, lines.size()); i++) 
        {
            System.out.println("Line " + i + ": " + lines.get(i));
        }
        
        // Skip the header line
        for (int i = 1; i < lines.size(); i++) 
        {  // Start from index 1 to skip header
            String line = lines.get(i).trim();
            if (!line.isEmpty()) 
            {
                try {
                    // Handle both comma and period as decimal separators
                    String normalizedLine = line.replace(",", ".");
                    double value = Double.parseDouble(normalizedLine);
                    responseList.add(value);
                } catch (NumberFormatException e) 
                {
                    System.err.println("Warning: Invalid number format at line " + (i + 1) + ": '" + line + "'");
                    System.err.println("Attempting to parse after removing non-numeric characters...");
                    try 
                    {
                        // Try parsing after removing all non-numeric characters except decimal point
                        String cleanedLine = line.replaceAll("[^0-9.]", "");
                        if (!cleanedLine.isEmpty()) 
                        {
                            double value = Double.parseDouble(cleanedLine);
                            responseList.add(value);
                            System.err.println("Successfully parsed as: " + value);
                        }
                    } catch (NumberFormatException e2) {
                        System.err.println("Warning: Skipping line " + (i + 1) + ": Unable to parse as number");
                    }
                }
            }
        }
        
        if (responseList.isEmpty()) {
            throw new IOException("No valid data found in file: " + filePath);
        }
        
        // Print summary of read data
        System.out.println("\nSuccessfully read " + responseList.size() + " response values");
        System.out.println("First few values:");
        for (int i = 0; i < Math.min(5, responseList.size()); i++) {
            System.out.printf("Value %d: %.2f%n", i + 1, responseList.get(i));
        }
        
        double[] responseArray = new double[responseList.size()];
        for (int i = 0; i < responseArray.length; i++) {
            responseArray[i] = responseList.get(i);
        }
        return responseArray;
    }

    public static void main(String[] args) {
        try {
            String predictorsFile = "/Users/yashpreetsingh/Downloads/real_estate_price_size_year_2_utf8.csv";
            String responseFile = "/Users/yashpreetsingh/Downloads/sheet1.csv";
    
            System.out.println("Reading predictor variables...");
            System.out.println("From file: " + predictorsFile);
            double[][] X = readPredictors(predictorsFile);
            System.out.println("Predictors loaded: " + X.length + " observations with " + X[0].length + " variables");
            
            // Print first few rows of predictors
            System.out.println("\nFirst 5 rows of predictors:");
            for (int i = 0; i < Math.min(5, X.length); i++) {
                System.out.println("Row " + (i+1) + ": " + Arrays.toString(X[i]));
            }
            
            System.out.println("\nReading response variable...");
            System.out.println("From file: " + responseFile);
            double[] y = readResponse(responseFile);
            System.out.println("Response variable loaded: " + y.length + " observations");
            
            // Print first few response values
            System.out.println("\nFirst 5 response values:");
            for (int i = 0; i < Math.min(5, y.length); i++) {
                System.out.println("Response " + (i+1) + ": " + y[i]);
            }
    
            // Verify data alignment
            if (X.length != y.length) {
                System.err.println("\nERROR: Data mismatch!");
                System.err.println("Predictor file has " + X.length + " rows");
                System.err.println("Response file has " + y.length + " rows");
                System.err.println("Please ensure both files contain the same number of observations.");
                throw new IllegalArgumentException("Number of observations doesn't match between predictors (" + 
                    X.length + ") and response (" + y.length + ")");
            }
    
            // Truncate predictor data to match response data if necessary
            if (X.length > y.length) {
                double[][] truncatedX = new double[y.length][];
                System.arraycopy(X, 0, truncatedX, 0, y.length);
                X = truncatedX;
            }
    
            // Fit the model
            multivariate_regression1 model = new multivariate_regression1(X, y);
    
            // Print model statistics
            System.out.println("\nRegression Results:");
            System.out.println("Intercept (β₀): " + model.intercept());
            double[] coefficients = model.coefficients();
            System.out.println("Size coefficient (β₁): " + coefficients[0]);
            System.out.println("Year coefficient (β₂): " + coefficients[1]);
            System.out.println("R-squared: " + model.R2());
    
            // Calculate slopes
            System.out.println("\nSlopes:");
            System.out.println("Price change per square foot: $" + String.format("%.2f", coefficients[0]));
            System.out.println("Price change per year: $" + String.format("%.2f", coefficients[1]));
    
            // Calculate discriminant (R²)
            double discriminant = model.R2();
            System.out.println("\nDiscriminant (R²): " + String.format("%.4f", discriminant));
            System.out.println("This means " + String.format("%.1f%%", discriminant * 100) + 
                              " of the variance in price is explained by size and year");
    
            // Interactive prediction section
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("\nPredict house price (enter negative values to exit)");
                System.out.print("Enter house size (square feet): ");
                double size = scanner.nextDouble();
                if (size < 0) break;
    
                System.out.print("Enter year built: ");
                double year = scanner.nextDouble();
                if (year < 0) break;
    
                // Make prediction
                double[] features = {size, year};
                double predictedPrice = model.predict(features);
    
                System.out.println("\nPrediction Results:");
                System.out.println("House Size: " + String.format("%.0f", size) + " sq ft");
                System.out.println("Year Built: " + String.format("%.0f", year));
                System.out.println("Predicted Price: $" + String.format("%,.2f", predictedPrice));
    
                // Calculate confidence metrics
                double meanSize = Arrays.stream(X).mapToDouble(row -> row[0]).average().orElse(0);
                double meanYear = Arrays.stream(X).mapToDouble(row -> row[1]).average().orElse(0);
                double distanceFromMean = Math.sqrt(
                    Math.pow((size - meanSize)/meanSize, 2) + 
                    Math.pow((year - meanYear)/meanYear, 2)
                );
    
                System.out.println("\nPrediction Confidence Metrics:");
                if (distanceFromMean > 2.0) {
                    System.out.println("Warning: Input values are far from training data mean - prediction may be less reliable");
                } else if (distanceFromMean > 1.0) {
                    System.out.println("Note: Input values are moderately far from training data mean");
                } else {
                    System.out.println("Input values are within typical range of training data");
                }
            }
    
            scanner.close();
    
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}