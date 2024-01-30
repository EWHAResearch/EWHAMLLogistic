# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import pandas.util.testing as tm
import seaborn as sns
import missingno as mno

# Load the dataset
df = pd.read_csv('AE_data.csv', index_col='VAERS_ID') # Load the file organizing VAERS AE data

# Fill empty cells with 0
df = df.fillna(0) 

# Splitting the dataset into independent (x) and dependent (y) variables
x = df.loc[:, 'AGE>65':'HTN'] # Select 41 variable columns of prediction model
y = df['MI_AE'] # Select one of the five serious AEFIs (AF_AE, AKI_AE, CVA_AE, MI_AE, PE_AE) we are interested in

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Normalize (scale) the data
from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply oversampling using SMOTE
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state = 0)
os_data_X, os_data_Y = os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data = os_data_X, columns=x.columns)
os_data_Y= pd.DataFrame(data=os_data_Y,columns=['MI_AE']) # Select one of the five serious AEFIs (AF_AE, AKI_AE, CVA_AE, MI_AE, PE_AE)

# Display the number of cases in oversampled data
print("Number of no MI case in oversampled data",len(os_data_Y[os_data_Y['MI_AE']==0])) # Select one of the five serious AEFIs (AF_AE, AKI_AE, CVA_AE, MI_AE, PE_AE)
print("Number of MI case",len(os_data_Y[os_data_Y['MI_AE']==1])) # Select one of the five serious AEFIs (AF_AE, AKI_AE, CVA_AE, MI_AE, PE_AE)


# Extract the features selected by RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 1000)
rfe = RFE(logreg, n_features_to_select = 60, step = 1)
rfe = rfe.fit(os_data_X, os_data_Y.values.ravel())

# Extract the features selected by RFE
new_cols = [];
for index, b in enumerate(rfe.support_):
  print(b)
  if b == True:
    new_cols.append(x.columns[index])

# Display the selected features
print(new_cols)

# Create the final dataset with selected features
final_X = os_data_X[new_cols]
final_y = os_data_Y['MI_AE'] # Select one of the five serious AEFIs (AF_AE, AKI_AE, CVA_AE, MI_AE, PE_AE)

# Fit the logistic regression model
import statsmodels.api as sm
logit_model = sm.Logit(final_y, final_X)
result = logit_model.fit()

# Display the logistic regression summary
result_summary_1 = result.summary2()
print(result_summary_1)

# Calculate and display the odds ratios
params = result.params
conf = result.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
result_summary_2 = np.exp(conf)
print(result_summary_2)

# Draw the ROC curve
from sklearn import metrics
from matplotlib.backends.backend_pdf import PdfPages

def draw_roc_curve(X_test, y_test, logreg):
  from sklearn.metrics import roc_auc_score
  from sklearn.metrics import roc_curve
  logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
  fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
  fig = plt.figure()
  plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  fig.savefig('/content/drive/MyDrive/Colab_Notebooks/Log_ROC_adverse_event.pdf')
  plt.show()

# Specify whether to use the original test set or create a new one
use_original_testset = True


if use_original_testset == True:
   # Use the original test set
  X_test_df = pd.DataFrame(data = X_test, columns=x.columns)
  X_test_df_simp = X_test_df[new_cols]

  # Fit the logistic regression model on the final dataset
  logreg = LogisticRegression(max_iter = 500)
  logreg.fit(final_X, final_y)

  # Predictions on the simplified test set
  y_pred = logreg.predict(X_test_df_simp)

  # Evaluate the model
  from sklearn.metrics import confusion_matrix
  confusion_matrix = confusion_matrix(y_test, y_pred)
  print(confusion_matrix)

  from sklearn.metrics import classification_report
  print(classification_report(y_test, y_pred))

  # Draw ROC curve
  chart = draw_roc_curve(X_test_df_simp, y_test, logreg)

else:
  # Create a new test set
  X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(final_X, final_y, test_size=0.3, random_state=42)
  
  # Fit the logistic regression model on the new training set  
  logreg = LogisticRegression(max_iter = 500)
  logreg.fit(X_train_new, y_train_new)

  # Predictions on the new test set
  y_pred_new = logreg.predict(X_test_new)

  # Evaluate the model
  from sklearn.metrics import confusion_matrix
  confusion_matrix = confusion_matrix(y_test_new, y_pred_new)
  print(confusion_matrix)

  from sklearn.metrics import classification_report
  print(classification_report(y_test_new, y_pred_new))

  # Draw ROC curve
  draw_roc_curve(X_test_new, y_test_new, logreg)