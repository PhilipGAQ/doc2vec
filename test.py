import pickle
import os
import inspect
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
classifiers_path='models/outputs/test'
new_path = os.path.join(base_path, classifiers_path)
print(new_path)