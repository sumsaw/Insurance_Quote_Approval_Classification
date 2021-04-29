# Insurance_Quote_Approval_Classification
Supervised Binary Classification Problem to determine whether the customer will accept the qupte or not 

# Home Site Quite Conversion Challenge 
​
Before asking someone on a date or skydiving, it's important to know your likelihood of success. The same goes for quoting home insurance prices to a potential customer. Homesite, a leading provider of homeowners insurance, does not currently have a dynamic conversion rate model that can give them confidence a quoted price will lead to a purchase. 
​
Using an anonymized database of information on customer and sales activity, including property and coverage information, Homesite is challenging you to predict which customers will purchase a given quote. Accurately predicting conversion would help Homesite better understand the impact of proposed pricing changes and maintain an ideal portfolio of customer segments. 
​
## Main Challenges 
​
This dataset was huge ~260K rows( aka samples) and 298 (features) and to add to that challenge the data was anonymized so 
doing feature engineering would be very random and usually brute force . I though of handeling this via feature selection and boosting methodology 
​
__I implemented two feature selection stratergies__ 
​
- __Mutual information:__
Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
​
- __Reculsive Feature Elimination:__
Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
​
After inspecting and performing EDA on the selected features I decided to treat all featues as catergorical. 
​
Once I have the feature selected to 50 from 298 I triend two model one simple __Logistic regression__ with one-hot encoding and other __LightGBM__ . With logistic regression I Was able to get the ROC-AUC score to 0.95 but the model took a long time to train due to large number of one-hot encoding 
​
I hyper-parameter tuned two Light GBM model with __Optuna__. Optuna is a hyperparameter framework . One feature which I like about it is that it allows us to stop the run for un-promising combination of values . This allows us to run hyper-parameter search for a larger grid.  
​
First model was trained on features obtained using mutual information which gave the ROC-AUC score as 0.93 and the second model was trained with features obtained from RFE which gave me a ROC-AUC score of 0.96+  For the final private test submission I was able to get a score of 0.9627 on the private leader board. 
​
Finally I used Sklearn Pipeline to optimize the prediction workflow for the test set. This allowed me to skip storing all the feature encoding values for 50 feature columns. 
​
## Key Learning 
​
- Feature Selection Techniques 
- Sklearn Pipeline 
​
## Part1 Notebook 
I will also link to this notebook my work where I optimized and did some EDA on the dataset 
​
​
## Upvote if you like the work 
LinkedIn: https://www.linkedin.com/in/sawantsumeet/
