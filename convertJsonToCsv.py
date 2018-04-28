# Python 3.6 code

import sys
import pandas as pd # Python data analysis library
from pandas import DataFrame # 2-D tabular data structure with labeled axes
import json # light-weight data interchange format inspired by JavaScript

# Set path to input json file
data=r'/home/sp/multiPartyHRI/rawKinectData.txt'

# open file and load it as json
def js_r(data):
    with open(data, encoding='utf-8') as f_in:
        return(json.load(f_in))

if __name__ == "__main__":
    my_dic_data = js_r(data)
#    print("This is my dictionary", my_dic_data)

keys = my_dic_data.keys()
#print("the original dict keys", keys)
df = pd.DataFrame(my_dic_data)

# To separate one column while still keeping the original column
#df['ElbowLeftX'] = df.ElbowLeft.str[0]

# To separate one column while removing the original column
#df['ElbowLeftX'], df['ElbowLeftY'], df['ElbowLeftZ'] = zip(*df.pop('ElbowLeft'))

# Create a list of all features to be split
listToBeSplit = ['ElbowLeft', 'ElbowRight', 'Head', 'HipLeft', 'HipRight','Neck',
                'ShoulderLeft', 'ShoulderRight', 'SpineBase','SpineMid',
                'SpineShoulder', 'WristLeft', 'WristRight','pedes_pos']

# Split all the features in the above list into 3 columns: x, y & z
for i in range(len(listToBeSplit)):
    df[listToBeSplit[i]+'X'], df[listToBeSplit[i]+'Y'], df[listToBeSplit[i]+'Z'] = zip(*df.pop(listToBeSplit[i]))

# To save the dataframe as .csv file
df.to_csv('rawKinectData.csv')
