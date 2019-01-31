from __future__ import absolute_import, division, print_function
#-------------------------------------------------------------------
import os.path
import sys
# archivo CSV
if (len(sys.argv)!=3):
  print("Args missing: <training_file.csv> <testing_file.csv>")
  exit(1)
if (not os.path.isfile(sys.argv[1])):
  print("Can't read file %s",sys.argv[1])
  exit(2)
if (not os.path.isfile(sys.argv[2])):
  print("Can't read file %s",sys.argv[2])
  exit(2)
csv = sys.argv[1]
csvt = sys.argv[2]

#-------------------------------------------------------------------
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn import preprocessing
import h5py

#-------------------------------------------------------------------
def mapTextColumn(data,columna):
  data[columna] = data[columna].apply(lambda x: str(x))
  encoder = preprocessing.LabelEncoder()
  encoder.fit(data[columna])
  data[columna] = encoder.transform(data[columna])

def limitNumber(data,columna,maximo):
  data[columna] = data[columna].apply(lambda x: int(x) if (int(x)<maximo) else maximo)
  
def normalize(df,columna):
  max_c=df[columna].max()
  min_c=df[columna].min()
  if (max_c>min_c):
  	return (df[columna]-min_c)/(max_c-min_c)
  else:
  	return 1.0

def loadCSV(csvfile):
  # Leo CSV
  data = pd.read_csv(csvfile,";")
  return data

def saveCSV(df,csvfile):
  df.to_csv(csvfile,";")

def preprocessData(data):
  # Proceso datos
  data['fnac']=pd.to_numeric(data.fnac.str.slice(6, 10))

  # Mapeo datos de texto
  mapTextColumn(data,'sexo')
  mapTextColumn(data,'estudios')
  mapTextColumn(data,'actividad')
  mapTextColumn(data,'comuna')
  mapTextColumn(data,'plan')

  # Limito # reclamos (max=3)
  limitNumber(data,'recl_3',3.0)
  limitNumber(data,'recl_12',3.0)

  # Normalizo
  data['sexo'] = normalize(data,'sexo')
  data['estudios'] = normalize(data,'estudios')
  data['actividad'] = normalize(data,'actividad')
  data['comuna'] = normalize(data,'comuna')
  data['plan'] = normalize(data,'plan')
  data['fnac'] = normalize(data,'fnac')
  data['recl_3'] = normalize(data,'recl_3')
  data['recl_12'] = normalize(data,'recl_12')

  train_x = pd.DataFrame(data,columns=['sexo','estudios','actividad','comuna','plan','fnac','recl_3','recl_12'])
  train_y = pd.DataFrame(data,columns=['exited'])
  return (train_x,train_y)

#-------------------------------------------------------------------
# Loading dataset
churns_ly = loadCSV(csv)
(train_x,train_y) = preprocessData(churns_ly)
#-------------------------------------------------------------------
# XGBoost
gbm = xgb.XGBClassifier(max_depth=16, n_estimators=25, learning_rate=0.01).fit(train_x, train_y.values.ravel())
#-------------------------------------------------------------------
# Prediction
churns_ly_test_s = loadCSV(csvt)
churns_ly_test = loadCSV(csvt)
(test_x,test_y) = preprocessData(churns_ly_test)
predictions = gbm.predict(test_x)
churns_ly_test_s['prediction']=predictions
print(churns_ly_test_s)

saveCSV(churns_ly_test_s,'prediction.csv')

hit = 0
miss = 0
bad = 0
count = 0
exited = 0

for i in range(len(churns_ly_test_s)):
  count = count + 1
  if churns_ly_test_s.loc[i][9] == 1:
    exited = exited + 1
  if churns_ly_test_s.loc[i][10] == 1 and churns_ly_test_s.loc[i][9] == 1:
   hit = hit + 1
  elif churns_ly_test_s.loc[i][10] == 0 and churns_ly_test_s.loc[i][9] == 1:
   miss = miss + 1
  elif churns_ly_test_s.loc[i][10] == 1 and churns_ly_test_s.loc[i][9] == 0:
   bad = bad + 1
 
print("Exited = %d" % exited)
if (exited)>0:
  print("Hits = %.1d %% (%d/%d)" % (100*hit/exited, hit, exited))
  print("Misses = %.1d %% (%d/%d)" % (100*miss/exited, miss, exited))
  print("False hits= %.1d %% (%d/%d)" % (100*bad/(count-(exited)), bad, count-(exited)))
else:
  print("Hits = >100 %% (%d/%d)" % ( hit, exited))
  print("Misses = >100 %% (%d/%d)" % ( miss, exited))
  print("False hits= %.1d %% (%d/%d)" % (100*bad/(count-(exited)), bad, count-(exited)))

