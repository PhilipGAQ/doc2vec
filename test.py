import pickle
import os
config={'dm':1,'hs':1,"negative":0}
string=str(config['dm'])+str(config['hs'])+str(config['negative'])
classifiers_path='models/outputs/test'
string=str(config['dm'])+str(config['hs'])+str(config['negative'])
filename = os.path.join(classifiers_path, 'output_' + string + '.pkl')
data = {'name': 'John', 'age': 30}
with open(filename, 'wb') as file:
    pickle.dump(data, file)