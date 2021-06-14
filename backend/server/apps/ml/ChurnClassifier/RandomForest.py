import joblib
import pandas as pd
import numpy as np
# from pandas.core.arrays import categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
class RandomForest_Classifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        # self.endcoders = joblib.load(path_to_artifacts + "")
        self.model = joblib.load(path_to_artifacts + "randomforest.joblib")
        # chỉ cần lưu lại giá trị model, nhưng phần preprocessing sẽ định nghĩa thẳng trong def ở dưới
    def preprocessing(self,input_data):
        #Json to pandas Dataframe
        input_data = pd.DataFrame(input_data, index = [0])
        # fill missing value
        # input_data.fillna(self.values_fill_missing)
        #replace name:
        input_data['SeniorCitizen'] = input_data['SeniorCitizen'].replace({1:'Yes',0:'No'})
        #change numerial to categorial columns
        conditions1 = [
            (input_data['MonthlyCharges'] <= 30),
            (input_data['MonthlyCharges'] > 30) & (input_data['MonthlyCharges'] <= 70),
            (input_data['MonthlyCharges'] > 70) & (input_data['MonthlyCharges'] <= 99),
            (input_data['MonthlyCharges'] > 99)
            ]
        # create a list of the values we want to assign for each condition
        values1 = ['0-30', '30-70', '70-99', '99plus']
        # create a new column and use np.select to assign values to it using our lists as arguments
        input_data['montlycharge_group'] = np.select(conditions1, values1)

        conditions2 = [
            (input_data['tenure'] <= 20),
            (input_data['tenure'] > 20) & (input_data['tenure'] <= 40),
            (input_data['tenure'] > 40) & (input_data['tenure'] <= 60),
            (input_data['tenure'] > 60)
            ]
        # create a list of the values we want to assign for each condition
        values2 = ['0-20', '20-40', '40-60', '60plus']

        # create a new column and use np.select to assign values to it using our lists as arguments
        input_data['tenure_group'] = np.select(conditions2, values2)

        conditions3 = [
            (input_data['TotalCharges'] <= 2000),
            (input_data['TotalCharges'] > 2000) & (input_data['TotalCharges'] <= 4000),
            (input_data['TotalCharges'] > 4000) & (input_data['TotalCharges'] <= 6000),
            (input_data['TotalCharges'] > 6000)
            ]
        # create a list of the values we want to assign for each condition
        values3 = ['0-2k', '2k-4k', '4k-6k', '6kplus']

        # create a new column and use np.select to assign values to it using our lists as arguments
        input_data['totalcharges_group'] = np.select(conditions3, values3)

        #ID columns
        ID_col = ['customerID']
        # target columns
        target_col = ['Churn']
        # numerical columns
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # binary columns with 2 values:
        bin_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']

        # columns more than 2 values
        multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod',
                      'montlycharge_group', 'tenure_group', 'totalcharges_group']
        # encoder
        le = LabelEncoder()
        for i in bin_cols:
            input_data[i] = le.fit_transform(input_data[i])
        #Duplicating columns for multi value columns
        input_data = pd.get_dummies(data = input_data, columns = multi_cols)
        # scaling numberical categorial
        std = StandardScaler()
        scaled = std.fit_transform(input_data[num_cols])
        scaled = pd.DataFrame(scaled, columns = num_cols)
        # dropping original values merging scaled values of numerical columns
        input_data = input_data.drop(columns = num_cols, axis =1)
        input_data = input_data.merge(scaled, left_index = True, right_index = True, how = 'left')
        # drop columns
        input_data = input_data.drop(['customerID'], axis = 1)
        # drop no internet service columns
        if 'InternetService_No' in input_data.columns:
        
            if (input_data['InternetService_No'] == 1).any():

                input_data = input_data.drop(columns = ['DeviceProtection_No internet service','OnlineBackup_No internet service',
                                    'OnlineSecurity_No internet service','TechSupport_No internet service',
                                    'StreamingTV_No internet service','StreamingMovies_No internet service'], axis = 1)
        # fill other missing value from standard training input data
        trainmode = pd.DataFrame(self.values_fill_missing, index = [0])

        #construct column
        train_col = trainmode.columns
        input_col = input_data.columns

        # difference between each columns
        differences = train_col.difference(input_col)

        # set anonther difference column to zero
        input_data[differences] = 0

        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "No Churn"
        if input_data[1] > 0.5:
            label = "Churn"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            # print(prediction)
            prediction = self.postprocessing(prediction)
            # print(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
