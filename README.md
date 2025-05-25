# Amazon Product Rating Predictor
### 1.  Finish Major Preprocessing

The dataset we chose for our analysis was well cleaned and highly processed.  However, given the size of the entire dataset, which includes ~300 million review observations, the training dataset is partitioned before model runs to leverage parallel processing.  Given the limitations and computational cost needed to train on the entire dataset, we sub-sampled our dataset to begin the model testing process.

### 2.  First Model Train
Initially our intention was to attempt a prediction based on a linear regression model (generating predictions bewteen 0 and 5), but we found better results using a random forest model and decided to develop that model further.  We will continue to test different hyperparameter selections, as well as further processing of the dataset to highlight features that have stronger impact on accurate classification.

### 3.  Model Evaluation
Our Random Forest model fits well within the ideal zone of the fitting graph. The training and test accuracies are nearly identical (65.6%), indicating that the model is neither underfitting nor overfitting. It generalizes well to unseen data and falls near the optimal point on the bias-variance curve.

### 4.  Model Fit and Next Model Steps
Random Forest model runs using 1%, 5%, and 10% of the dataset produce similar results, yielding ~65% accuracy using 80% of each subset as train data and 20% as the test data.  Additionally we want to consider improvements on the linear regression model, if possible.  Additionally, we believe a logistic regression model could perform well.  However, due to the size of the dataset and the length of the text reviews, proper scalable solutions must be tested to succeed in this model type.  Vectorizing millions of review sentences requires thoughtful implementation.

### 5.  Milestone 3 Notebook
[Click here to see notebook](Notebook/Amazon_Reviews_3.2.ipynb)

### 6.  Conclusion
To address the question, "What is the conclusion of your first model? What can be done to possibly improve it?": Our first model, linear regression with rounded predictions, performed poorly because it was not designed for categorical outcomes. To improve upon that initial approach, we would focus on modelling methods better sutied for our dataset and the multi-class prediction we are developing. This is precisely what led us to explore Random Forests and other classifiers. At this point, we’re focused on tuning the Random Forest model further — adjusting hyperparameters such as the number of trees, maximum depth, and feature subsets — to optimize performance and reduce variance without overfitting.  Other models will also be considered that may lend themselves better to our classification goal.

### Setup
```bash
pip install -r requirements.txt
# or see top of `exploration_spark.ipynb` for !pip commands

