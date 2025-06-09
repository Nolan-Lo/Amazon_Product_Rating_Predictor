# Amazon Product Rating Predictor

### Abstract

This project leverages Amazon product reviews and item metadata to predict customer star ratings by applying sentiment analysis and structured data modeling. To handle the large-scale dataset efficiently, we use PySpark alongside the UCSD Supercomputer. A core component of our approach is understanding which features are significant in predicting customer sentiment, with the added opportunity to deep dive into the metadata such as product category, price, features, and verified purchase status. Beyond developing a prediction model, our goal is to identify features that can signal customer sentiment across various categories price points, and product types.  

We hope to gain a deeper understanding of consumer behavior, enabling businesses to optimize product offerings, improve marketing strategy, and align pricing with customer expectations. The combination of sentiment-driven insights and the metadata available offer powerful resources for enhancing decision-making across business operations.

### Link to Jupyter Notebook
[Click here to see notebook](https://github.com/Nolan-Lo/Amazon_Product_Rating_Predictor/blob/Milestone4/Notebook/Amazon_Reviews_M4.ipynb)

---

## Introduction

In our initial data exploration to determine the focus of our project, we discovered a robust dataset developed by the McAuley Lab research team at UC San Diego that compiled amazon customer review data.   This dataset is incredibly expansive - it includes over 300 million reviews, a rich set of categorical features for each review observation, and no null values for any of the features in the dataset.  

Given the opportunity to work with such a dataset, and our shared understanding of the proliferation of ecommerce and digital transactions, we decided to focus our efforts on predicting the star rating of Amazon reviews based on their features, such as review length, title, verified purchase status, and more.

We agreed that the ability to predict online customer reviews would yield significant commercial impact for businesses that want to expand or focus their source of sales online.  An accurate model would enable business leaders to enhance their online storefronts, and anticipate headwinds given the competitive nature of some product categories or consumer segments.

We believed the impressive quality and size of the dataset would give us a strong foundation to train our model.  

In addition to the dataset of amazon customer reviews, a corresponding dataset of product metadata was available.  This would enable further analysis into the feature significance and trends we identify during model development.


## Methods and Results

### 1. Data Exploration

We began by understanding the structure and distribution of our dataset, which consisted of over **300 million** product reviews:

**Total Reviews**: `300,260,938`

#### Schema Overview
![Nulls by columns](/images/schema.png)


#### Null Values
We observed **zero missing values** across all columns:
![Nulls by columns](/images/nulls_by_col.png)


#### Category Distribution
![Nulls by columns](/images/category_distribution.png)

This distribution shows that some categories are more prevelant in the dataset than others.


#### Rating Distribution
![Nulls by columns](/images/rating_distribution.png)

This distribution suggests a strong skew toward positive ratings.


#### Text Length Summary
![Nulls by columns](/images/text_length.png)

This reveals high variability in review length, which could influence model predictions and require normalization or capping during preprocessing.



### 2. Preprocessing

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



### 3. Model 1 – Linear Regression

We began with a **linear regression model** as a baseline.

#### Motivation:
  - **Computationally cheap**, allowing for quick tuning of parameters
  - Simple and easy to interpret

#### Experiments & Parameter Tuning

We experimented with many different custom rounding bins to improve accuracy and help mitigate the class imbalance of the rounded predictions. 

| Rounded Rating | Custom Bin Interval |
|----------------|---------------------|
| 1.0            | (-∞, 1.50)          |
| 2.0            | [1.50, 2.85)        |
| 3.0            | [2.85, 3.30)        |
| 4.0            | [3.30, 4.05)        |
| 5.0            | [4.05, ∞)           |

| Rating | True Count | Predicted Count |
|--------|------------|-----------------|
| 1.0    | 138,758    | 136,333         |
| 2.0    | 47,918     | 45,963          |
| 3.0    | 83,963     | 87,322          |
| 4.0    | 171,332    | 174,213         |
| 5.0    | 214,629    | 212,769         |

However it did not perform as well as we hoped. 

- **Reasons for poor performance**:
  - Assumes linearity, which doesn't capture complex interactions.
  - Unable to model nonlinear relationships between review characteristics and rating.
  - Product categories introduce heterogeneity, which the linear model couldn't adjust for effectively.



### 4. Model 2 – Random Forest

To better capture non-linear patterns and interactions in the data, we transitioned to a **Random Forest** model.

#### Motivation:
- Handles **categorical variables** more effectively.
- Captures **nonlinear dependencies**.
- More robust to outliers and noise.

#### Implementation & Experiments

We conducted several trials to assess the effect of sample size and model parameters on performance.

##### Sample Size Experiments and Results

| Sample Size | Accuracy Impact | Processing Time Impact |
|-------------|------------------|-------------------------|
| 1%          | 0.6564           | 266.54 seconds          |
| 5%          | 0.6558           | 583.98 seconds          |
| 10%         | 0.6559           | 1150.65 seconds         |

- We observed that compared to linear regression, random forest was significantly more accurate. However, within the different sampling sizes for random forest, **accuracy did not significantly improve** with larger samples, but **processing time increased drastically**.

##### Hyperparameter Tuning

- **Number of Trees**: 10  
- **Max Depth**: 5

These parameters were chosen to reduce runtime. Accuracy remained relatively stable when increasing these values, while runtime increased noticeably.

#### Link to notebook
The notebook containing all of the models can be found [here.](https://github.com/Nolan-Lo/Amazon_Product_Rating_Predictor/blob/Milestone4/Notebook/Amazon_Reviews_M4.ipynb)


## Discussion

### 1. Data Subsampling and Preprocessing
We first tried to train on the full ~300 million-record dataset but ran into memory errors and prohibitively long runtimes, even with a 20 percent subset. To address this, we sampled 1%(3 million), 5% (15 million), and 10% (30 million) of the data. The random forest achieved 65.64% accuracy on 1% data, 65.58% on 5% data, and 65.59% on 10% data. Training and test accuracies converged around 65.6 %, showing the model was neither underfitting nor overfitting. This consistency indicates that, beyond a modest subsample, simply adding more data did not improve performance under our current feature set.

### 2. Reflecting on the Regression-Based Model
We began with linear regression because of its simplicity and interpretability, rounding continuous outputs to discrete star ratings. We undersampled the 5.0 star rating as they accounted for almost half of the entire dataset. We began with traditional rounding, which had a low accuracy of around 30%. We then decided to tune our rounding parameters, transitioning to custom bins. This achieved 52.33% test accuracy, while a vast improvement, ultimately fell short of expectations. We also tried logistic regression and XGBoost on the same features, but saw no significant gains. Treating ratings as continuous ignored their categorical nature, so we shifted our focus to classification methods.

### 3. Interpreting the Random Forest Results
We chose a random forest classifier to handle high-dimensional, sparse features without extensive preprocessing. It scored about 65.6% accuracy across all sample sizes, with training and test accuracies closely matched. These results seem plausible for natural language data, yet the mid-sixties plateau suggests our features capture only surface-level signals.

### 4. Category Specific Modeling
Product categories on Amazon differ in vocabulary and rating patterns. A single model risks washing out these distinctions. Future work will involve training separate models for each major category, tailoring feature engineering and algorithm choice to the data volume and characteristics of each group.

### 5. Shortcomings
Our mid-sixties accuracy is credible for a first pass, but several limitations stand out:
 Feature engineering relied on basic term-frequency vectors rather than richer representations like TF-IDF, embeddings, or transformer encodings.
 A skew toward four- and five-star reviews biases predictions, suggesting class weighting or resampling may help.
Hyperparameter tuning was manual. An automated grid or Bayesian searches could find better settings.
Runtime limitations caused memory errors on larger subsets, highlighting the need for distributed training pipelines.


## Conclusion
What initially seemed like our dataset’s greatest feature turned out to be its most significant hurdle.  The sheer size of the dataset made it near impossible to train a model on anything more than a 10% subset, given our resource constraints.  

From the outset, we attempted to create a generalized prediction model. However, given the volume of data available, we now recognize that a more effective approach would involve building specialized models tailored to individual product categories. This would not only reduce dataset size per model, enabling faster iterations and hyperparameter tuning, but would also likely improve accuracy by capturing category-specific patterns. 

It is clear that if we focused our modelling efforts on specific categories, thereby subsetting the dataset into category relevant observations, we could speed up our ability to iterate and tune our hyperparameters.

Our final model was a Random Forest classifier trained on a 10% sample of the dataset, or approximately 30 million rows. This model achieved an accuracy of **65.59%**, which is a strong result considering we were predicting one of **six possible labels** (ratings from 0 through 5). 

Ultimately, while our generalized model delivered reasonable predictive performance, the structure and richness of the dataset present a clear opportunity to boost performance by shifting toward category-specific modeling strategies.


## Statement of Collaboration

Nolan Lo. Data Scientist. I contributed to all parts of the project. We primarily worked, collaborated and discussed as a team on all parts of the project. We did not have a designated team leader, but I did help with creating the github, scheduling, and starting discussions of how work and responsibilities will be allocated. For milestone 1 we all collaborated on the abstract, bouncing ideas off each other. We ended up using my final draft of the abstract, but it was a culmination of all our ideas and drafts. For milestone 2  I contributed to our data analysis tables and plots, and combined, aggregated, and submitted our final github branch. For milestone 3 I created the random forest model that we ended up using as our final model. For milestone 4 we met and allocated which sections would be written by who. The majority of my work for the writeup can be found in the methods and results section, specifically the data analysis, and random forest model, results, and observations. 

Matt Zidell.  Data Scientist.  We primarily worked, collaborated and discussed as a team on all parts of the project.  I discovered the Amazon dataset and presented it to the team as a feasible foundation for our project.  I successfully loaded the dataset to the SDSC and configured my repository so that my teammates could access the dataset for data exploration and model training. I contributed data exploration analysis for Milestone 2 and I helped summarize the results of the random forest model for Milestone 3.  For milestone 4, I worked alongside my teammates to organize our results and draft a report that would convey our research in a concise and compelling manner.

Mehul. Data Scientist. I contributed to all parts of the project. I primarily worked on creating an initial Linear Regression model and gave constructive feedback on the write-ups.

Jacob. Data Scientist. We as a group each contributed to all parts of the project, breaking down every large task and each taking on a responsibility to complete that part. For milestone 2, I helped with some of the data exploration, creating some of the visualizations that are seen in the notebook. For milestone 3 I worked on creating a linear regression model to fit the data, and helped merge our separate notebooks for the final turn in. For milestone 4, I worked alongside the team to plan out our final write up and helped write the methods and discussion pertaining to the linear regression model.

Brian. Data Scientist. We worked as a group contributing to every part of the project. For milestone 2, I created the visuals for change of ratings as time goes by. For milestone 3, I worked on creating a model using xgboost and also incorporating methods such as undersampling, oversampling, and dividing the train/test split in different ways to enhance accuracy(But the accuracy was too low to be included in the write up). For milestone 4, I worked with our team to create the final write up.



