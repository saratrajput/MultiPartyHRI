# Python 3.6 code

import sys # Provides access to variables and functions used by interpreter
import pandas as pd # Python data analysis library
from pandas import DataFrame # 2-D tabular data structure with labeled axes
import json # light-weight data interchange format inspired by JavaScript

from sys import argv # For the list of command line arguments passed
script, inputFile, outputFile = argv
# Set path to input json file
#data=r'/home/sp/multiPartyHRI/rawKinectData.txt'

# open file and load it as json
def json_read(inputFile):
    with open(inputFile, encoding='utf-8') as f_in:
        return(json.load(f_in))

#if __name__ == "__main__":
my_dic_data = json_read(inputFile)

#keys = my_dic_data.keys()
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

########## Cleaning the data ########## 
# Dropping face size since it's a column of zeros
df.drop('face_size', axis=1, inplace=True)

# Dropping First column of indexes since it is redundant
# For some reason the previous command didn't work for this column, so this:
# df.drop(df.columns[1], axis=1, inplace=True)

# Mapping 'face_engaged': no = 0; yes = 1
# With yes and no: face_looking away, face_engaged

yesNoList = ['face_engaged', 'face_lookingaway'] # List of columns with only yes and no

for i in range(len(yesNoList)):
    df[yesNoList[i]] = df[yesNoList[i]].map( {'no': 0, 'yes': 1, 'unknown': 0} ).astype(int)
    
# Mapping 'face_glasses': unknown & no = 0; yes = 1 
# List of columns with unknown, no and yes: 
unknownYesNoList = ['face_glasses', 'face_happy', 'face_lefteyeclosed', 'face_mouthmoved',
                            'face_mouthopen', 'face_righteyeclosed']
for i in range(len(unknownYesNoList)):
    df[unknownYesNoList[i]] = df[unknownYesNoList[i]].map( {'no': 0, 'yes': 1, 'unknown': 0} ).astype(int)

# To save the dataframe as .csv file
df.to_csv(outputFile + '.csv')
