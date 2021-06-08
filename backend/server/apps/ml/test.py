from django.test import TestCase
from apps.ml.ChurnClassifier.logit import Logit_Classifier
from apps.ml.registry import MLRegistry
import inspect

class MLTests(TestCase):
    def test_logit_algorithm(self):
        input_data = {
            "customerID":290865,
            "gender":"Female",
            "SeniorCitizen":1,
            "Partner":"Yes",
            "Dependents":"No",
            "tenure":0,
            "PhoneService":"Yes",
            "MultipleLines":"Yes",
            "InternetService":"DSL",
            "OnlineSecurity":"Yes",
            "OnlineBackup":"Yes",
            "DeviceProtection":"Yes",
            "TechSupport":"Yes",
            "StreamingTV":"Yes",
            "StreamingMovies":"Yes",
            "Contract":"One year",
            "PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check",
            "MonthlyCharges":400,
            "TotalCharges": 30000,
        }

        my_alg = Logit_Classifier()
        response = my_alg.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('No Churn', response['label'])

    # add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "ChurnClassifier"
        algorithm_object = Logit_Classifier()
        algorithm_name = "logit"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Ian"
        algorithm_description = "Logistic Classifier with simple pre- and post-processing"
        algorithm_code = inspect.getsource(Logit_Classifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)