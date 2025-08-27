## Health Risk Assessment – BMI Classification
## Project Overview

This project builds a BMI (Body Mass Index) Classifier that predicts a person’s health category (Extremely Weak → Extreme Obesity) based on Gender, Height, and Weight.
We use PyTorch for deep learning, scikit-learn for preprocessing, and Seaborn/Matplotlib for visualization.

## Dataset

File: 500_Person_Gender_Height_Weight_Index.csv

## Features:

Gender → Male / Female

Height → in centimeters

Weight → in kilograms

Index → Health Category (0-5)

Index	Category
0	Extremely Weak
1	Weak
2	Normal
3	Overweight
4	Obesity
5	Extreme Obesity
## Workflow
## 1. Data Preprocessing

Gender encoded → Male=1, Female=0

Features normalized with MinMaxScaler

Train-test split (80%-20%)

## 2. Model Architecture (PyTorch)

Input Layer: 3 neurons (Gender, Height, Weight)

Hidden Layers: [16 → 32 → 16] with ReLU + Dropout

Output Layer: 6 neurons (corresponding to BMI classes)

Loss Function: CrossEntropyLoss

Optimizer: Adam (lr=0.001)

Training: 10,000 epochs

## 3. Evaluation

Metrics: Accuracy, Classification Report, Confusion Matrix

Visualization: Scatter plot of Height vs Weight grouped by BMI category

## 4. Prediction on New Data

Example:

new_sample = [[1, 170, 70]]  # Male, 170cm, 70kg
Predicted Category → "Normal"


Training for 10,000 epochs achieved high accuracy.

Model can predict BMI class given gender, height, and weight.

Scatter plot example:

X-axis: Height

Y-axis: Weight

Color: BMI Index

## How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn







## Check model evaluation, charts, and predictions.
