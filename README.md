# Project Datasets: Diamonds and Census Income

This repository contains two separate projects focusing on data analysis: one on the Diamonds dataset and the other on the Census Income dataset. Each project is structured into multiple parts, including analysis, preprocessing, and modelling. Follow the instructions below to run each part of the projects.

## How to Run

### General Instructions
1. **Navigate to the Program Directory:**
   - Open a terminal window.
   - Navigate to the directory containing the specific program using the `cd` command, either `diamonds` or `census_income`. For example:
     ```bash
     cd census-income-and-diamonds-datasets/diamonds
     ```

### Diamonds Dataset

#### Running Specific Parts
- Once inside the Diamonds project directory, you have the option to run specific parts of the program using command-line arguments. The available options are `--preprocessing`, `--modelling` and `--all`.

- **To run the preprocessing, use:**
  ```bash
  python main.py --preprocessing

- **To run the modelling, use:**
  ```bash
  python main.py --modelling

- **To run both parts in sequence, use:**
  ```bash
  python main.py --all

### Census Income Dataset

#### Running Specific Parts
- Once inside the Diamonds project directory, you have the option to run specific parts of the program using command-line arguments. The available options are `--analysis`, `--preprocessing`, `--modelling` and `--all`.

- **To run the analysis, use:**
  ```bash
  python main.py --analysis

- **To run the preprocessing, use:**
  ```bash
  python main.py --preprocessing

- **To run the modelling, use:**
  ```bash
  python main.py --modelling

- **To run both parts in sequence, use:**
  ```bash
  python main.py --all

### Additional Notes

- Ensure that you have all the necessary libraries and dependencies installed in your Python environment. You can install them using pip install requirements.txt from the parent directory.
- Warnings from libraries have been suppressed for cleaner output, but please make sure to check for any critical warnings or errors in your environment or console.
- All plots generated during analysis and modelling will be saved in the corresponding Plots directory.
All tables generated during analysis will be saved as separate sheets in the analysis.xlsx file located in the corresponding Tables directory.
The table generated during modelling will be saved in the model_comparison_results.xlsx file located in the corresponding Tables directory.