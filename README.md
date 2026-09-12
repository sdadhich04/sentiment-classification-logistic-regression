# Sentiment Classification with Logistic Regression

This repository contains a Jupyter notebook that classifies product reviews as positive or negative. It removes neutral reviews, converts review text to bag-of-words counts, trains logistic-regression models, evaluates predictions with a majority-class baseline and a manually constructed confusion matrix, inspects influential words and review probabilities, and compares several L2 regularization strengths.

The notebook was developed in the context of **CSE 416 (Introduction to Machine Learning)** at the University of Washington.

## Tools and libraries

- Python 3 and Jupyter
- pandas, NumPy, scikit-learn, Matplotlib, and Seaborn
- No hardware is required.

## Data and limitations

The notebook expects a file named `food_products.csv` in the repository directory. The file is not included in this repository. The code uses its `rating`, `review`, and `summary` columns, and cannot run from a fresh clone until a compatible file is supplied.

## Running

```bash
python -m pip install -r requirements.txt
```

Place `food_products.csv` beside `sentiment_classification_logistic_regression.ipynb`, open the notebook in Jupyter, and run the cells from top to bottom.

## Credits

Code and notebook curation: **Sparsh Dadhich**.
