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

### Link to Jupyter Notebook
[Click here to see notebook](https://github.com/Nolan-Lo/Amazon_Product_Rating_Predictor/blob/main/Notebook/Amazon_Reviews_Final.ipynb)

### Setup
```bash
pip install -r requirements.txt
# or see top of `exploration_spark.ipynb` for !pip commands

