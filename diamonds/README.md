# Diamonds Dataset

## How to Run

This program is structured into multiple parts, including analysis, preprocessing, and modelling. Follow the instructions below to run each part:

1. Navigate to the Program Directory:
   - Open a terminal window.
   - Navigate to the directory containing the program using the 'cd' command. For example:
    ```
    cd path/to/program_directory
    ```

2. Running Specific Parts:
   - Once inside the program directory, you have the option to run specific parts of the program using command-line arguments. The available options are `--preprocessing`, `--modelling` and `--all`. 
   
   - To run the preprocessing, use:
    ```
    python main.py --preprocessing
    ```
   
   - To run the modelling, use:
    ```
    python main.py --modelling
    ```
    Note: If the cleaned data file does not exist in the 'Data/Cleaned' directory, you will be prompted to run preprocessing first.
   
   - To run both parts in sequence, use the `--all` option:
    ```
    python main.py --all
    ```

Note: Ensure that you have all the necessary libraries and dependencies installed in your Python environment before running the program. You can usually install them using pip.

Warnings from libraries have been suppressed for cleaner output, but please make sure to check for any critical warnings or errors in your environment or in the console.

Additional Notes: 
   - All plots generated during analysis and modelling will be saved in the 'Plots' directory located in the parent directory of the program.
   - All tables generated during analysis will be saved as seperate sheets in the analysis.xlsx file located in the parent directory of the program.
   - The table generated during modelling will be saved in the regression_results.xlsx file located in the parent directory of the program.
