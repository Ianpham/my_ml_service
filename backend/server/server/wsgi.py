"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()


# ML Registry Lưu ý đường dẫn server.apps có cần không vì server cùng chung level vs apps
import inspect
from apps.ml.ChurnClassifier.logit import Logit_Classifier
from apps.ml.ChurnClassifier.RandomForest import RandomForest_Classifier
from apps.ml.registry import MLRegistry

try:
    registry = MLRegistry() # create MLRegistry

    # churnclassifier
    churn = Logit_Classifier()
    # add to MLRegistry
    registry.add_algorithm(endpoint_name = "ChurnClassifier",
                            algorithm_object = churn,
                            algorithm_name = "logit",
                            algorithm_status = "production",
                            algorithm_version = "0.0.1",
                            owner = "Ian",
                            algorithm_description = "Logistic Classifier with simple pre- and post-processing",
                            algorithm_code = inspect.getsource(Logit_Classifier))

    # random forest classifier
    rf = RandomForest_Classifier()

    # add to registry
    registry.add_algorithm(endpoint_name = "ChurnClassifier",
                        algorithm_object = rf,
                        algorithm_name = "RandomForest",
                        algorithm_status = "testing",
                        algorithm_version = "0.0.1",
                        owner = "Ian",
                        algorithm_description = "Random Forest Classifier with simple pre- and post-processing",
                        algorithm_code = inspect.getsource(RandomForest_Classifier))
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))