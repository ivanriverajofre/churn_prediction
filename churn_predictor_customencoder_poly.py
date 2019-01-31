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
#Domain Values
glo={}
glo['sexo']=["F","M"]
glo['estudios']=["POSTGRADO","SUPERIOR","TECNICA","MEDIA","BASICA","SIN_ESTUDIOS"]
glo['comuna']=["Providencia","Las Condes","Vitacura","Lo Barnechea","San Miguel","La Reina","Ñuñoa","Santiago","Macul","Maipú","La Florida","Estación Central","San Joaquín","Quilicura","Peñalolén","Cerrillos", "La Granja","Quinta Normal","Lampa","Huechuraba","Independencia","La Cisterna","Lo Prado","Pudahuel","Padre Hurtado","Puente Alto","Conchalí","Renca","San Bernardo","Recoleta","Colina","Lo Espejo","El Bosque","San Ramón","Melipilla","Cerro Navia","Buin","Pedro Aguirre Cerda","Paine","Peñaflor","Talagante","La Pintana"]
glo['actividad']=["Profesional Independiente","Oficio independiente","Profesional Dependiente","Estudiante","Oficio dependiente","Trabajo no remunerado","Jubilado/Pensionado","Cesante"]
glo['plan']=["PLAN_70000","PLAN_60000","PLAN_50000","PLAN_45000","PLAN_40000","PLAN_35000","PLAN_30000","PLAN_25000","PLAN_20000","PLAN_15000","PLAN_10000","PLAN_5000"]
#-------------------------------------------------------------------

def mapTextColumn(data,columna):
  data[columna] = data[columna].apply(lambda x: glo[columna].index(x))

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

  # Relaciono atributos
  data['np1'] = data['actividad']*data['plan']
  data['np2'] = data['sexo']*data['fnac']
  data['np3'] = data['recl_3']*data['recl_12']
  data['np4'] = data['estudios']*data['actividad']
  data['np5'] = data['comuna']
  data['np6'] = data['plan']*data['plan']
  data['np7'] = data['sexo']

  # Normalizo
  data['np1'] = normalize(data,'np1')
  data['np2'] = normalize(data,'np2')
  data['np3'] = normalize(data,'np3')
  data['np4'] = normalize(data,'np4')
  data['np5'] = normalize(data,'np5')
  data['np6'] = normalize(data,'np6')
  data['np7'] = normalize(data,'np7')

  train_x = pd.DataFrame(data,columns=['np1','np2','np3','np4','np5','np6','np7','np8'])
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

