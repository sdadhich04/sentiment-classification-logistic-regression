# Sentiment Classification with Logistic Regression

Classifying product reviews as positive or negative sentiment using
bag-of-words features and logistic regression — including manual
performance evaluation (not just calling `.score()`), coefficient-based
interpretation of which words drive predictions, and a study of how L2
regularization strength trades off training vs. validation accuracy.

Built from coursework for **CSE 416 (Introduction to Machine Learning)**,
University of Washington. See [Results](#results) for the actual numbers
this notebook produced.

## What this does

1. Filters out neutral (rating = 3) reviews, keeping only clearly positive
   or negative examples, and vectorizes review text with `CountVectorizer`.
2. Establishes a majority-class baseline classifier to contextualize what
   "good" accuracy actually means for this class balance.
3. Trains an unregularized logistic regression model, then manually
   extracts the most positive and most negative words by directly reading
   off the largest/smallest learned coefficients.
4. Finds the most positive and most negative individual reviews using
   `predict_proba`.
5. Manually computes true positives/false positives/false negatives/true
   negatives from boolean masks (rather than only calling a library
   metric), to build the confusion matrix from first principles.
6. Sweeps L2 regularization strength across 7 values (`solver="saga"`,
   `fit_intercept=False`), building a full train-vs-validation accuracy
   table to visualize the underfitting/overfitting tradeoff.

## Results

- After filtering neutral reviews: **889** labeled reviews remain.
- Unregularized logistic regression: most negative coefficient **−5.99**
  ("not"), most positive coefficient **+11.02** ("great") — sensible,
  human-interpretable sentiment words.
- Regularization sweep (train / validation accuracy):
  - λ = 0.01: **0.8805 / 0.6966** — low regularization, high train accuracy
    but a visible generalization gap.
  - λ = 100,000: **0.5021 / 0.5169** — regularized down to essentially
    baseline (majority-class-level) performance on both sets.
- A confusion-matrix heatmap and a coefficient-path plot (how individual
  word coefficients shrink as regularization increases) are saved as
  outputs in the notebook.

The regularization sweep is the clearest result here: it shows the
classic accuracy curve where light regularization overfits to the small
(889-review) training set, and heavy regularization erases the model's
learned signal entirely — with the actual sweet spot visible in between.

## What's original vs. course-provided

The course notebook provided `CountVectorizer` setup and a
`plot_confusion_matrix` helper. Written for this assignment: the majority
baseline, the coefficient-based most-positive/most-negative word
extraction, the most-positive/most-negative review lookup, the manual
TP/FP/FN/TN computation, and the full L2 regularization sweep loop
(building both the coefficient table and the accuracy table).

## Known limitations

- **Data not included.** The notebook loads `food_products.csv`, a product
  reviews dataset used in this course (in the spirit of the "amazon_baby"
  sentiment dataset from the Coursera Machine Learning Specialization); the
  raw file isn't available to include here. To rerun, supply a matching
  CSV with review text and rating columns in the same directory.

## Running this code

Requires Python with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and
`seaborn`. Open `sentiment_classification_logistic_regression.ipynb` in
Jupyter, supply `food_products.csv` in the same directory, and run all
cells top to bottom.
