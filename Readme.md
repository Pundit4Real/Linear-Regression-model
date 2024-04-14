# Linear Regression Models

This repository contains code for two linear regression models:
1. **Multiple Linear Regression Model for Profit Prediction**
2. **Linear Regression Model for Exam Score Prediction**

## Prerequisites
- Python 3
- Jupyter Notebook or any Python IDE
- Create a virtual environment ( recommended )

## Installation
1. Create a virtual environment

        python -m venv venv 
    
2. Clone or download this repository to your local machine.

            https://github.com/Pundit4Real/Linear-Regression-model.git

3. Activate the virtual environment:

- On Windows:

            venv\Scripts\activate

- On macOS and Linux:

            source venv/bin/activate

4. Install the required Python libraries using pip by running the command below:

            pip install -r requirements.txt

5. Ensure you have the datasets '50_Startups.csv' and 'students_scores.csv' in the same directory as the code.

## Running the Models
1. Open a terminal or command prompt.

2. Navigate to the directory containing the code files.

3. To run the Multiple Linear Regression Model for Profit Prediction:
- Run the script `multiple_linear_regression_profit.py` using Python:

                  python multiple_linear_regression_profit.py
  

4. To run the Linear Regression Model for Exam Score Prediction:
- Run the script `linear_regression_exam_scores.py` using Python:
  
          python linear_regression_exam_scores.py
  

5. Both scripts will train the respective models, make predictions, display evaluation metrics, and visualize the results.

## Customization
- You can modify the code to use your own datasets by replacing the provided dataset files.
- Experiment with different parameters and preprocessing techniques to improve model performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
