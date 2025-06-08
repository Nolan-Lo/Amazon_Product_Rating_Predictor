# Amazon Product Rating Predictor

### Abstract

This project leverages Amazon product reviews and item metadata to predict customer star ratings by applying sentiment analysis and structured data modeling. To handle the large-scale dataset efficiently, we use PySpark alongside the UCSD Supercomputer. A core component of our approach is natural language processing, where sentiment analysis is used to extract emotional and opinion-based indicators from the review text and titles. These sentiment signals, combined with structured metadata such as product category, price, features, and verified purchase status, form the basis of machine learning models that predict the numerical star ratings left by customers. Beyond prediction, our goal is to uncover why certain products receive the reviews they do, and to identify patterns that vary across categories, price points, and product types. This includes analyzing how product attributes correlate with sentiment and perceived value. Our findings provide a deeper understanding of consumer behavior, enabling businesses to optimize product offerings, improve marketing strategy, and align pricing with customer expectations. The combination of sentiment-driven insights and metadata analysis offers a powerful tool for enhancing decision-making across business operations.


### Data Preprocessing

1. **Loading Data**
   -  Load .parquet files from a directory, with each file representing a different product category. 
   -  A category column is added to keep track of the source.
  
2. **Combining Datasets**
   -  All the individual category files are merged into a single Spark DataFrame using unionByName, ensuring columns are aligned.
  
3. **Handling Missing Data**
   - Check each column for missing values, including special handling for numeric fields that may contain NaNs.
     
4. **Review Text**  
   - Use Spark ML’s `RegexTokenizer` -> `StopWordsRemover` -> `CountVectorizer` or `TFIDF`  
   - Handle missing text via empty string

5. **Metadata**  
   - Impute missing `price` with median or remove all instances if there is a small amount. 
   - One-hot encode `main_category` & `verified_purchase` & `helpful_vote` with `StringIndexer` + `OneHotEncoder`

6. **Handle Class Imbalance**
   - Undersample majority class(5 star rating) & oversample minority class(1, 2, 3, 4 star rating)

7. **Avoid Data Leakage**
   - When creating train/val/test split make sure that the split is in chronological order


## Milestone 3
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
To address the question, "What is the conclusion of your first model? What can be done to possibly improve it?": Our first model, linear regression with rounded predictions, performed poorly because it was not designed for categorical outcomes. To improve upon that initial approach, we would focus on modelling methods better suited for our dataset and the multi-class prediction we are developing. This is precisely what led us to explore Random Forests and other classifiers. At this point, we’re focused on tuning the Random Forest model further — adjusting hyperparameters such as the number of trees, maximum depth, and feature subsets — to optimize performance and reduce variance without overfitting.  Other models will also be considered that may lend themselves better to our classification goal.


### Link to Jupyter Notebook
[Click here to see notebook](https://github.com/Nolan-Lo/Amazon_Product_Rating_Predictor/blob/main/Notebook/Amazon_Reviews_Final.ipynb)

### Setup
```bash
pip install -r requirements.txt
# or see top of `exploration_spark.ipynb` for !pip commands

