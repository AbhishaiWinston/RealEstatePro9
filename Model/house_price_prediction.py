import pandas as pd
import numpy as np
import math
from math import sqrt
data = pd.read_csv("bengaluru_house_prices.csv")
data.dropna(inplace=True)
data['size'] =data['size'].apply(lambda x: x.split(' ')[0])
data['size'].unique()




# merge phase road etc
pattern = r'\d+(st|nd|rd|th)\s'
def extract_location(location):
    location = re.sub(pattern, '', location)  # Remove patterns like "1st", "2nd", etc.
    location = re.sub(r'\b(phase|sector|road|zone|layout|avenue|colony)\b.*$', '', location, flags=re.IGNORECASE)  # Remove specified terms and anything after them
    return location.strip()
data['location'] = data['location'].apply(lambda x: extract_location(x))


# In[9]:


len(data['location'].unique()) # length before spelling correction


# In[10]:


from collections import Counter
from fuzzywuzzy import fuzz


# In[11]:


def identify_misspellings(locations, known_locations, threshold=80):
    misspellings = {}

    for location in locations:
        found = False
        for known_location in known_locations:
            # Calculate Levenshtein distance
            similarity = fuzz.ratio(location, known_location)
            if similarity > threshold:
                found = True
                misspellings[location] = known_location
                break
        if not found:
            misspellings[location] = None  # Mark as not found

    return misspellings


# Geting known locations

# In[12]:


import requests
from bs4 import BeautifulSoup

# Wikipedia URL
url = "https://en.wikipedia.org/wiki/List_of_neighbourhoods_in_Bangalore"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table", {"class": "wikitable"})


# In[13]:


known_names = []
for table in tables:
   
    neighborhood_names = []
    for row in table.find_all("tr")[1:]:
        name = row.find("td").text.strip()
        neighborhood_names.append(name)

    for name in neighborhood_names:
        known_names.append(name)


# In[14]:


misspelled_loc = identify_misspellings(data['location'].unique(),known_names)


# In[15]:


for misspelled, corrected in misspelled_loc.items():
    if corrected:
        data['location'] = data['location'].replace(misspelled, corrected)


# In[16]:


len(data['location'].unique())# after correction


# ## Exploratory Data Analysis
# 

# In[17]:


from sklearn.preprocessing import LabelEncoder
import scipy.stats as ss


# In[18]:


# Convert categorical variables to numerical labels
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

corr_matrix = data_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# ### Distribution Of Features in Data Columns

# In[19]:


plt.figure(figsize=(30, 18), dpi=200)

plt.subplot(2,3,1)
sns.histplot(data["price"], linewidth=0,kde=True)

plt.subplot(2,3,2)
sns.histplot(data["bath"], linewidth=0,kde=True)

plt.subplot(2,3,3)
sns.histplot(data["balcony"], linewidth=0,kde=True)

plt.subplot(2,3,4)
sns.histplot(data["total_sqft"], linewidth=0,kde=True)

plt.subplot(2,3,5)
sns.histplot(data["size"], linewidth=0,kde=True)

plt.suptitle("Distribution of Data column wise", fontsize=20)
plt.show()


# ### Plotting Features column vs Target column

# In[20]:


data['size'] =data['size'].apply(lambda x: int(x))

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        if '-' in value:
            lower, upper = map(float, value.split('-'))
            return (lower + upper) / 2
        else:
            return None  

data['total_sqft'] = data['total_sqft'].apply(convert_to_float)


# In[21]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.regplot(x=data["bath"], y=data["price"], scatter_kws={"color": "green"}, line_kws={"color": "red"})
plt.subplot(2,2,2)
sns.regplot(x=data["total_sqft"], y=data["price"], scatter_kws={"color": "green"}, line_kws={"color": "red"})
plt.subplot(2,2,3)
sns.regplot(x=data["balcony"], y=data["price"], scatter_kws={"color": "green"}, line_kws={"color": "red"})
plt.subplot(2,2,4)
sns.regplot(x=data["size"], y=data["price"], scatter_kws={"color": "green"}, line_kws={"color": "red"})


# In[22]:


data.info()


# In[23]:




# In[24]:


data.drop(['society',"availability"],axis=1, inplace=True)


# ## Remove Outliers

# In[25]:


def remove_outliers(df, cols, threshold=1.5):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Specify numerical columns to check for outliers
numerical_cols = ['size', 'total_sqft', 'bath', 'balcony', 'price']

# Remove outliers from the specified numerical columns
data = remove_outliers(data, numerical_cols)


# ## Encoding

# In[26]:


df = data.copy()


# In[27]:


location_encoded={}
location_means = df.groupby('location')['price'].mean()

df['location_encoded'] = df['location'].map(location_means)
location_encoded[str(df['location'])]=df['location'].map(location_means)
df.drop('location', axis=1, inplace=True)


# In[28]:


area_type_encoded={}
area_type_means = df.groupby('area_type')['price'].mean()
df['area_type_encoded'] = df['area_type'].map(area_type_means)
area_type_encoded[str(df['area_type'])]=df['area_type'].map(area_type_means)
df.drop('area_type', axis=1, inplace=True)


# In[29]:


df.info()


# In[30]:


df.head()


# ## Linear models

# ### Split into test and train

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


df.columns


# In[33]:


input_cols = ['size', 'total_sqft', 'bath', 'balcony', 'location_encoded','area_type_encoded']

target_cols = 'price'

input = df[input_cols].copy()
target = df[target_cols].copy()


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state = 7)


# In[35]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[36]:


pd.DataFrame(X_train, columns=X_train.columns).plot.box(figsize=(20,5), rot=90)
plt.show()


# In[37]:


mm = MinMaxScaler().fit(X_train)
X_train_mm = mm.transform(X_train)
X_train_mm = pd.DataFrame(X_train_mm, columns=X_train.columns)
X_test_mm = mm.transform(X_test)
X_test_mm = pd.DataFrame(X_test_mm, columns=X_test.columns)
X_train_mm.plot.box(figsize=(20,5), rot=90)
plt.show()


# In[38]:


ss = StandardScaler().fit(X_train)
X_train_ss = ss.transform(X_train)
X_train_ss = pd.DataFrame(X_train_ss, columns=X_train.columns)
X_test_ss = ss.transform(X_test)
X_test_ss = pd.DataFrame(X_test_ss, columns=X_test.columns)
X_train_ss.plot.box(figsize=(20,5), rot=90)
plt.show()


# ## Model Training

# In[39]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score


# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.svm import SVR


# ### LinearRegression

# In[41]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))

# lr = LinearRegression()
lr.fit(X_train_ss, y_train)
y_predlrss = lr.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_predlrss))

# lr = LinearRegression()
lr.fit(X_train_mm, y_train)
y_predlrmm = lr.predict(X_test_mm)
print('R2- SCORE(Minmaxscaled):', metrics.r2_score(y_test,y_predlrmm))

r2_lr = metrics.r2_score(y_test, y_pred)
r2_lr_ss = metrics.r2_score(y_test, y_predlrss)
r2_lr_mm = metrics.r2_score(y_test, y_predlrmm)

# Create a list of lists for the data
new_data = [
    ["LinearRegression", r2_lr,r2_lr_mm,r2_lr_ss]
]

columns = ['model','R2',"R2 -MinMaxScaler","R2 - StandardScaler"]
results = pd.DataFrame(new_data, columns=columns)


# In[64]:


results.head()


# ### KNN

# In[47]:


from sklearn.neighbors import KNeighborsRegressor


# In[43]:


rmse_val = []
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train_mm, y_train)
    pred=model.predict(X_test_mm)
    error = sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
curve = pd.DataFrame(rmse_val)

rmse_val1 = []
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train_ss, y_train)
    pred=model.predict(X_test_ss)
    error = sqrt(mean_squared_error(y_test,pred))
    rmse_val1.append(error)
curve1 = pd.DataFrame(rmse_val1)

print('Orange and Blue depict RSME for MinMaxScaler and Standard Sacalr')
plt.figure(figsize=(12,7))
plt.plot(curve)
plt.plot(curve1)
plt.show()


# ### Tree Based

# In[44]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import tree
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# ## DecisionTree

# In[45]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
# Define the parameter grid for grid search
param_grid = {
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Create the GridSearchCV object
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='r2')

# Perform the grid search on the original dataset (X_train, y_train)
grid_search.fit(X_train, y_train)

# Get the best estimator and its corresponding R2 score
best_dt = grid_search.best_estimator_
best_dt_r2 = grid_search.best_score_
print("Best Parameters:", grid_search.best_params_)

y_pred_dt = best_dt.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred_dt))

best_dt.fit(X_train_mm, y_train)
y_pred_dtmm = best_dt.predict(X_test_mm)
print('R2- SCORE(Minmaxscaled):', metrics.r2_score(y_test,y_pred_dtmm))

best_dt.fit(X_train_ss, y_train)
y_pred_dtss = best_dt.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_dtss))

r2_dt = metrics.r2_score(y_test, y_pred_dt)
r2_dt_ss = metrics.r2_score(y_test, y_pred_dtss)
r2_dt_mm = metrics.r2_score(y_test, y_pred_dtmm)

new_row = pd.DataFrame({
    'model': ["DecisionTree"],
    'R2': [r2_dt],
    'R2 -MinMaxScaler': [r2_dt_mm],
    'R2 - StandardScaler': [r2_dt_ss]
})

results = pd.concat([results, new_row], ignore_index=True)
results.head()


# ## Ensamble Learning

# ## Voting

# In[49]:


from sklearn.neighbors import KNeighborsClassifier


# In[50]:


param_grid_dt = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'max_features': [1.0, 'sqrt']
}

param_grid_xg = {
    'learning_rate': [0.5, 0.7, 1.0],
    'n_estimators': [50, 100, 200]
}

grid_search_dt = GridSearchCV(DecisionTreeRegressor(), param_grid_dt, cv=5)
grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5)
grid_search_xg = GridSearchCV(xgb.XGBRegressor(verbosity=0), param_grid_xg, cv=5)

grid_search_dt.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_xg.fit(X_train, y_train)

best_dt = grid_search_dt.best_estimator_
best_rf = grid_search_rf.best_estimator_
best_xg = grid_search_xg.best_estimator_

print("Best Parameters dt:", grid_search_dt.best_params_)
print("Best Parameters rf:", grid_search_rf.best_params_)
print("Best Parameters xg:", grid_search_xg.best_params_)

vr = VotingRegressor([('dt', best_dt), ('knn', knn), ('lr', lr), ('rf', best_rf), ('xg', best_xg), ('gbr', gbr), ('etr', etr)])

vr.fit(X_train, y_train)
y_pred_vr = vr.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test, y_pred_vr))

grid_search_dt.fit(X_train_mm, y_train)
grid_search_rf.fit(X_train_mm, y_train)
grid_search_xg.fit(X_train_mm, y_train)

best_dt = grid_search_dt.best_estimator_
best_rf = grid_search_rf.best_estimator_
best_xg = grid_search_xg.best_estimator_
vr = VotingRegressor([('dt', best_dt), ('knn', knn), ('lr', lr), ('rf', best_rf), ('xg', best_xg), ('gbr', gbr), ('etr', etr)])

vr.fit(X_train, y_train)
y_pred_dtmm = best_dt.predict(X_test_mm)
print('R2- SCORE(Minmaxscaled):', metrics.r2_score(y_test,y_pred_dtmm))

grid_search_dt.fit(X_train_ss, y_train)
grid_search_rf.fit(X_train_ss, y_train)
grid_search_xg.fit(X_train_ss, y_train)

best_dt = grid_search_dt.best_estimator_
best_rf = grid_search_rf.best_estimator_
best_xg = grid_search_xg.best_estimator_
vr = VotingRegressor([('dt', best_dt), ('knn', knn), ('lr', lr), ('rf', best_rf), ('xg', best_xg), ('gbr', gbr), ('etr', etr)])

y_pred_dtss = best_dt.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_dtss))


r2_vr = metrics.r2_score(y_test, y_pred_vr)
r2_vr_ss = metrics.r2_score(y_test, y_pred_dtss)
r2_vr_mm = metrics.r2_score(y_test, y_pred_dtmm)

new_row = pd.DataFrame({
    'model': ["Voting Regressor"],
    'R2': [r2_vr],
    'R2 -MinMaxScaler': [r2_vr_mm],
    'R2 - StandardScaler': [r2_vr_ss]
})

results = pd.concat([results, new_row], ignore_index=True)
results.head()


# ### Random Forest

# In[51]:


rf = RandomForestRegressor(n_estimators= 200, max_depth = 15, max_features='sqrt')
rf.fit(X_train, y_train);
y_pred_rf = rf.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred_rf))

rf.fit(X_train_mm, y_train);
y_pred_rfmm = rf.predict(X_test_mm)
print('R2- SCORE(minmaxscaled):', metrics.r2_score(y_test,y_pred_rfmm))

rf.fit(X_train_ss, y_train);
y_pred_rfss = rf.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_rfss))

r2_rf = metrics.r2_score(y_test, y_pred_rf)
r2_rf_ss = metrics.r2_score(y_test, y_pred_rfss)
r2_rf_mm = metrics.r2_score(y_test, y_pred_rfmm)

new_row = pd.DataFrame({
    'model': ["Random Forest"],
    'R2': [r2_rf],
    'R2 -MinMaxScaler': [r2_rf_mm],
    'R2 - StandardScaler': [r2_rf_ss]
})

results = pd.concat([results, new_row], ignore_index=True)
results.head()


# In[ ]:


rf = RandomForestRegressor(n_estimators= 200, max_depth = 15, max_features='sqrt')
rf.fit(X_train, y_train)
import joblib
joblib.dump(rf, 'regression_model.joblib')


# ### ExtraTreesRegressor

# In[76]:


param_grid_etr = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'max_features': [1.0, 'sqrt', None]
}

grid_search_etr = GridSearchCV(ExtraTreesRegressor(random_state=0), param_grid_etr, cv=5)
grid_search_etr.fit(X_train, y_train)
best_etr = grid_search_etr.best_estimator_

y_pred_etr = best_etr.predict(X_test)

grid_search_etr.fit(X_train_ss, y_train)
best_etr = grid_search_etr.best_estimator_
y_pred_etr_mm = best_etr.predict(X_test_mm)

grid_search_etr.fit(X_train_ss, y_train)
best_etr = grid_search_etr.best_estimator_
y_pred_etr_ss = best_etr.predict(X_test_ss)

r2_etr = metrics.r2_score(y_test, y_pred_etr)
r2_etr_mm = metrics.r2_score(y_test, y_pred_etr_mm)
r2_etr_ss = metrics.r2_score(y_test, y_pred_etr_ss)


new_row = pd.DataFrame({
    'model': ["ExtraTreesRegressor"],
    'R2': [r2_etr],
    'R2 - MinMaxScaler': [r2_etr_mm],
    'R2 - StandardScaler': [r2_etr_ss]
})


results = pd.concat([results, new_row], ignore_index=True)
results.head()


# ### GradientBoostingRegressor

# In[77]:


for lr in [0.1,0.3,0.5,0.8,1]:
  model= GradientBoostingRegressor(learning_rate=lr)
  model.fit(X_train, y_train)
  print("Learning rate : ", lr, " Train score : ", model.score(X_train,y_train), " Test score : ", model.score(X_test,y_test))


# In[78]:


gbr = GradientBoostingRegressor(learning_rate=.1)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred_gbr))

gbr.fit(X_train_mm, y_train)
y_pred_gbrmm = gbr.predict(X_test_mm)
print('R2- SCORE(MinMaxScaler):', metrics.r2_score(y_test,y_pred_gbrmm))

gbr.fit(X_train_ss, y_train)
y_pred_gbrss = gbr.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_gbrss))

r2_gbr= metrics.r2_score(y_test, y_pred_gbr)
r2_gbr_ss = metrics.r2_score(y_test, y_pred_gbrss)
r2_gbr_mm = metrics.r2_score(y_test, y_pred_gbrmm)

new_row = pd.DataFrame({
    'model': ["GradientBoostingRegressor"],
    'R2': [r2_gbr],
    'R2 -MinMaxScaler': [r2_gbr_mm],
    'R2 - StandardScaler': [r2_gbr_ss]
})

results = pd.concat([results, new_row], ignore_index=True)
print(results)


# ### XG Boost

# In[79]:


xg = xgb.XGBRegressor(learning_rate = .5, n_estimators=50, verbosity = 0)
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred_xg))

xg.fit(X_train_mm, y_train)
y_pred_xgmm = xg.predict(X_test_mm)
print('R2- SCORE(MinMaxScaler):', metrics.r2_score(y_test,y_pred_xgmm))

xg.fit(X_train_ss, y_train)
y_pred_xgss = xg.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_xgss))

r2_xg= metrics.r2_score(y_test, y_pred_xg)
r2_xg_ss = metrics.r2_score(y_test, y_pred_xgss)
r2_xg_mm = metrics.r2_score(y_test, y_pred_xgmm)

new_row = pd.DataFrame({
    'model': ["XG Boost"],
    'R2': [r2_xg],
    'R2 -MinMaxScaler': [r2_xg_mm],
    'R2 - StandardScaler': [r2_xg_ss]
})

results = pd.concat([results, new_row], ignore_index=True)
print(results)


# In[80]:


results.reset_index(inplace=True)
results.set_index('model', inplace=True)
sorted_df = results.sort_values(by=results.columns.tolist(), axis=0, ascending=False)
print(sorted_df)


# In[82]:


results


# ## Visualize performance

# In[83]:


dt = DecisionTreeRegressor()
knn = KNeighborsRegressor(n_neighbors=100)
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators= 100, max_depth = 15, max_features='sqrt')
xg = xgb.XGBRegressor(learning_rate = .7, n_estimators=100, verbosity = 0)
gbr = GradientBoostingRegressor(learning_rate=.5)
etr = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)

vr = VotingRegressor([('dt', dt), ('knn', knn), ('lr', lr), ('rf', rf), ('xg', xg), ('gbr', gbr), ('etr', etr)])
vr.fit(X_train_ss, y_train);
y_pred_vrss = vr.predict(X_test_ss)


# In[84]:


pred_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_vrss})

# Plot using Seaborn
plt.figure(figsize=(8, 8))
sns.scatterplot(data=pred_df, x='True Values', y='Predicted Values', color='blue', alpha=0.25)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.title('True vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[85]:


from yellowbrick.regressor import ResidualsPlot

# Create a ResidualsPlot
visualizer = ResidualsPlot(vr)

# Fit the training data to the visualizer
visualizer.fit(X_train, y_train)

# Evaluate the model on the test data
visualizer.score(X_test, y_test)

# Draw residuals plot
visualizer.show()


# In[86]:


from yellowbrick.model_selection import FeatureImportances

estimator_importances = {}
for name, estimator in vr.named_estimators_.items():
    print(estimator)
    if hasattr(estimator, 'feature_importances_'):
        # Store feature importances
        estimator_importances[name] = estimator.feature_importances_
        # Create and show FeatureImportances visualizer
        visualizer = FeatureImportances(estimator, labels=X_train.columns)
        visualizer.fit(X_train, y_train)
        visualizer.show()
    else:
        print(f"Estimator {name} does not support feature importances.")


# ### Predicting Values

# In[93]:


results


# In[99]:


import random
import warnings
warnings.filterwarnings("ignore")
                        
rand = random.randint(0, len(X_test) - 1)

print("Features:")
print(X_test.iloc[rand])
print("Actual Target Value:")
print(y_test.iloc[rand])

lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression Prediction:")
print(lr.predict([X_test.iloc[rand]]))

dt = DecisionTreeRegressor(max_depth= 10, min_samples_leaf= 4, min_samples_split= 10)
dt.fit(X_train_ss, y_train)
print("Decision tree Prediction:")
print(dt.predict([X_test.iloc[rand]]))


rf = RandomForestRegressor(n_estimators= 200, max_depth = 15, max_features='sqrt')
rf.fit(X_train, y_train);
print("Random Forest Prediction:")
print(rf.predict([X_test.iloc[rand]]))

etr = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
print("Extra Trees Prediction:")
print(etr.predict([X_test.iloc[rand]]))

gbr = GradientBoostingRegressor(learning_rate=.1)
gbr.fit(X_train_mm, y_train)
print("Gradient Boosting Prediction:")
print(gbr.predict([X_test.iloc[rand]]))

xg = xgb.XGBRegressor(learning_rate = .5, n_estimators=50, verbosity = 0)
xg.fit(X_train, y_train)
print("XGBoost Prediction:")
print(xg.predict([X_test.iloc[rand]]))

vr = VotingRegressor([('dt', dt), ('knn', knn), ('lr', lr), ('rf', rf), ('xg', xg), ('gbr', gbr), ('etr', etr)])
vr.fit(X_train, y_train)
print("Voting Regressor Prediction:")
print(vr.predict([X_test.iloc[rand]]))


# In[ ]:




