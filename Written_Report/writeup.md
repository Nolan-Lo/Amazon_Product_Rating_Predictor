# Amazon Rating Predictor










## 1. Data Exploration

We began by understanding the structure and distribution of our dataset, which consisted of over **300 million** product reviews:

**Total Reviews**: `300,260,938`

### Schema Overview
![Nulls by columns](../images/schema.png)

### Null Values
We observed **zero missing values** across all columns:
![Nulls by columns](../images/nulls_by_col.png)

### Rating Distribution
![Nulls by columns](../images/rating_distribution.png)
This distribution suggests a strong skew toward positive ratings.

### Text Length Summary
![Nulls by columns](../images/text_length.png)
This reveals high variability in review length, which could influence model predictions and require normalization or capping during preprocessing.


## 2. Preprocessing

Given the massive size of the dataset and performance constraints, we implemented the following preprocessing strategies:

- **Sampling**: We initially attempted to train on the full dataset but encountered severe memory and runtime issues. As a solution, we:
  - Sampled **1%**, then **5%**, and finally **10%** of the dataset.
  - Found that accuracy remained fairly **constant across samples**, while computational time increased exponentially.
- **Feature Selection**:
  - Used features such as `text_len`, `category`, `verified_purchase`, and `helpful_vote` as predictors.
  - Encoded categorical features like `category`.
  - Avoided using full `text` or `images` fields directly due to dimensionality and noise.
- **Target Variable**:
  - Predicted `rating` as a regression task.

---

## 3. Model 1 – Linear Regression

We began with a **linear regression model** as a baseline due to its simplicity and interpretability. However, performance was limited:

- **Reasons for poor performance**:
  - Assumes linearity, which doesn't capture complex interactions.
  - Unable to model nonlinear relationships between review characteristics and rating.
  - Product categories introduce heterogeneity, which the linear model couldn't adjust for effectively.

---

## 4. Model 2 – Random Forest

To better capture non-linear patterns and interactions in the data, we transitioned to a **Random Forest** model.

### Motivation:
- Handles **categorical variables** more effectively.
- Captures **nonlinear dependencies**.
- More robust to outliers and noise.

### Implementation & Experiments

We conducted several trials to assess the effect of sample size and model parameters on performance.

#### Sample Size Experiments

| Sample Size | Accuracy Impact | Processing Time Impact |
|-------------|------------------|-------------------------|
| 1%          | 0.6564           | 266.54 seconds          |
| 5%          | 0.6558           | 583.98 seconds          |
| 10%         | 0.6559           | 1150.65 seconds         |

- We observed that compared to linear regression, random forest was significantly more accurate. However, within the different sampling sizes for random forest, **accuracy did not significantly improve** with larger samples, but **processing time increased drastically**.

#### Hyperparameter Tuning

- **Number of Trees**: 10  
- **Max Depth**: 5

These parameters were chosen to reduce runtime. Accuracy remained relatively stable when increasing these values, while runtime increased noticeably.

### Link to notebook
The notebook containing all of the models can be found [here](https://github.com/Nolan-Lo/Amazon_Product_Rating_Predictor/blob/Milestone4/Notebook/Amazon_Reviews_M4.ipynb)
