#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


# Load the dataset
Diabetesdf = Path("diabetes.csv")
df = pd.read_csv(Diabetesdf)
df.head()


# In[3]:


# Define the threshold for diabetes classification based on 'glyhb' levels
diabetes_threshold = 6.5
df['diabetes'] = (df['glyhb'] >= diabetes_threshold).astype(int)


# In[4]:


df


# In[5]:


# Drop rows with missing target variable 'glyhb'
df = df.dropna(subset=['glyhb'])


# In[8]:


# List of columns to be included in the modeling (excluding id and target variable)
feature_columns = ['chol', 'stab.glu', 'hdl', 'ratio', 'Age', 'Gender', 'Height', 'Weight 1', 'frame', 'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn']


# In[9]:


# Separate the features and target variable
X = df[feature_columns]
y = df['diabetes']


# In[10]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Preprocessing for numerical features: fill missing values with mean and scale them
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


# In[12]:


# Preprocessing for categorical features: fill missing values with 'missing' and one-hot encode
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[13]:


# Create preprocessor with Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[14]:


# Create a pipeline that processes the data and then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])


# In[15]:


# Train the model
pipeline.fit(X_train, y_train)


# In[16]:


# Predict on the test set
y_pred = pipeline.predict(X_test)


# In[17]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.5f}')


# In[18]:


# Check if the model meets the 75% accuracy requirement
if accuracy >= 0.75:
    print("The model meets the accuracy requirement.")
else:
    print("The model does not meet the accuracy requirement. Further optimization is needed.")


# In[19]:


# Define a hyperparameter grid for logistic regression
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],  # Regularization strength
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']  # Different algorithms
}


# In[21]:


# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)


# In[22]:


# Perform the grid search over the parameters
grid_search.fit(X_train, y_train)


# In[23]:


# Convert the grid search results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)


# In[24]:


results


# In[25]:


# Save the results to a CSV file
results.to_csv('grid_search_results.csv', index=False)


# In[26]:


# Evaluate the best model found by grid search on the test set
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)


# In[27]:


# Calculate metrics
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
classification_rep = classification_report(y_test, y_pred_optimized)


# In[28]:


# Output the results
best_params = grid_search.best_params_
best_score = grid_search.best_score_


# In[29]:


# Print the overall model performance
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validated Score: {best_score:.4f}')
print(f'Test Accuracy: {optimized_accuracy:.4f}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:




