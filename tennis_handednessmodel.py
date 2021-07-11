# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing import normalize

import scipy.stats as stats
import statsmodels.api as sm
from lowess import lowess
import statsmodels

!pip install lowess

!pip install statsmodels

def preprocess(years=list(range(1968,2021)),competitions=["futures","challengers","best"]):
    full_df = None
    title ={"best":"",
            "challengers":"_qual_chall",
            "futures":"_futures"}
    for year in years:
      for comp in competitions:
        file_name = "atp_matches"+title[comp]+"_"+str(year)+".csv"
        if full_df is None:
          full_df = pd.read_csv(file_name)
        else:
          full_df = pd.concat([full_df,pd.read_csv(file_name)])
    return full_df

data = preprocess([1968,
1969,
1970,
1971,
1972,
1973,
1974,
1975,
1976,
1977,
1978,
1979,
1980,
1981,
1982,
1983,
1984,
1985,
1986,
1987,
1988,
1989,
1990,
1991,
1992,
1993,
1994,
1995,
1996,
1997,
1998,
1999,
2000,
2001,
2002,
2003,
2004,
2005,
2006,
2007,
2008,
2009,
2010,
2011,
2012,
2013,
2014,
2015,
2016,
2017,
2018,
2019,
2020,
2021],['best'])
data.shape



pd.unique(data["winner_ht"])

data["winner_ht"].value_counts()

left_right_win = data[['winner_ht', 'loser_ht']]

left_right_win = left_right_win.loc[(left_right_win['winner_ht']!='NaN') & (left_right_win['loser_ht']!='NaN')]

left_right_win

left_right_win.describe()

left_right_win = left_right_win.loc[(left_right_win['winner_hand']=="R") & (left_right_win['loser_hand']=="L") | (left_right_win['winner_hand']=="L") & (left_right_win['loser_hand']=="R") | (left_right_win['winner_hand']=="R") & (left_right_win['loser_hand']=="R") | (left_right_win['winner_hand']=="L") & (left_right_win['loser_hand']=="L")]

left_right_win = left_right_win.fillna(left_right_win.mean())
left_right_win

#left_right_win['winner_hand'].value_counts()
left_right_win.loc[(left_right_win['winner_hand']=='L') | (left_right_win['loser_hand']=='L')]

sum = 303 +282
sum

a = 303/sum
a

a*100

data

def cleaning(dataset, columns):
  if "hand" not in columns:
    columns.append("hand")
  dataset = dataset.loc[(dataset['winner_hand']!='U') & (dataset['loser_hand']!='U')]
  all_cols = ["winner_" + col for col in columns] + ["loser_" + col for col in columns]
  data = dataset[all_cols]
  data = data[(data['winner_hand']=="R") & (data['loser_hand']=="L") | (data['winner_hand']=="L") & (data['loser_hand']=="R")]

  # fill the NaN values
  data = data.fillna(data.mean())
  data = pd.get_dummies(data, drop_first=True)
  data['winner_hand'] = data['winner_hand_R'].astype(float) # convert uint8 to float
  data['loser_hand'] = data['loser_hand_R'].astype(float)   # convert uint8 to float
  for col in columns:
    data[(col+'_dif')] = data[('winner_'+col)] - data[('loser_'+col)]

  flips = np.random.randint(0, 2, size=data.shape[0])

  data['winner'] = flips

  for col in columns:
    # xW + (W-1)*x
    data[(col+'_dif_rand')] = data[(col+'_dif')]* data['winner'] + (data['winner'] -1 )* data[(col+'_dif')]
  dif_cols = [col+"_dif_rand" for col in columns] +['winner']
  model_data = data[dif_cols]

  #normalizing
  x = model_data.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  model_data = pd.DataFrame(x_scaled)
  model_data.columns = dif_cols
  return model_data, data

model_data, full_data = cleaning(data, ['hand','age','ht','rank'])

model_data

df = data[['winner_rank', 'loser_rank']]
df = df.dropna()

df['dif'] = df['winner_rank'] - df['loser_rank']
df.reset_index()

df.iloc[169661]

df2 = df.loc[(df['dif'] >= 1) & (df['dif'] <= 2)]
df2.reset_index()

df2.iloc[1686]

0.093373 - 0.092006

x = df2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2 = pd.DataFrame(x_scaled)
df2.iloc[1686]

full_data

"""1392 row num"""

full_data.loc[(full_data['age_dif'] >= 1) & (full_data['age_dif'] <= 1.01)]

X_train, X_test, y_train, y_test = train_test_split(model_data.drop('winner', axis=1), model_data['winner'])
model_logreg = LogisticRegression(penalty = 'none')
model_logreg.fit(X_train, y_train)
print(X_train)

prediction = model_logreg.score(X_test, y_test)
coeffs = model_logreg.coef_
print(prediction)
print(coeffs)

"""# ***Making graphs***"""

def preprocess(years=list(range(1981,2021)),competitions=["futures","challengers","best"]):
  full_df = None
  title ={"best":"",
          "challengers":"_qual_chall",
          "futures":"_futures"}
  for year in years:
    for comp in competitions:
      file_name = "atp_matches"+title[comp]+"_"+str(year)+".csv"
      if full_df is None:
        full_df = pd.read_csv(file_name)
      else:
        full_df = pd.concat([full_df,pd.read_csv(file_name)])
  return full_df

def cleaning(dataset, columns):
  if "hand" not in columns:
    columns.append("hand")
  dataset = dataset.loc[(dataset['winner_hand']!='U') & (dataset['loser_hand']!='U')]
  all_cols = ["winner_" + col for col in columns] + ["loser_" + col for col in columns]
  data = dataset[all_cols]
  data = data[(data['winner_hand']=="R") & (data['loser_hand']=="L") | (data['winner_hand']=="L") & (data['loser_hand']=="R")]

  # fill the NaN values
  data = data.fillna(data.mean())
  data = pd.get_dummies(data, drop_first=True)
  data['winner_hand'] = data['winner_hand_R'].astype(float) # convert uint8 to float
  data['loser_hand'] = data['loser_hand_R'].astype(float)   # convert uint8 to float
  for col in columns:
    data[(col+'_dif')] = data[('winner_'+col)] - data[('loser_'+col)]

  flips = np.random.randint(0, 2, size=data.shape[0])
  data['winner'] = flips

  for col in columns:
    # xW + (W-1)*x
    data[(col+'_dif_rand')] = data[(col+'_dif')]* data['winner'] + (data['winner'] -1 )* data[(col+'_dif')]
  dif_cols = [col+"_dif_rand" for col in columns] +['winner']
  model_data = data[dif_cols]

  #normalizing
  x = model_data.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  model_data = pd.DataFrame(x_scaled)
  model_data.columns = dif_cols
  return model_data, data


def logreg(model_data):
  X_train, X_test, y_train, y_test = train_test_split(model_data.drop('winner', axis=1), model_data['winner'])
  model_logreg = LogisticRegression(penalty = 'none')
  model_logreg.fit(X_train, y_train)

  prediction = model_logreg.score(X_test, y_test)
  coeffs = model_logreg.coef_
  return coeffs.tolist()[0]

#takes in all previous functions
def allfunc(years=list(range(1968,2021)),competitions=["futures","challengers","best"], columns = ['hand','age','ht', 'rank', 'best_of']):
  data = preprocess(years, competitions)
  model_data, full_data = cleaning(data, columns)
  return logreg(model_data)

allfunc(years=[2020], competitions = ['best'],columns = ['hand','age','ht','rank'])

all_coefs = []
atp_year_range = []
chall_year_range = []
fut_year_range = []

year_range = list(range(1968,2021))
for year in year_range:
  try:
    all_coefs.append(allfunc(years=[year], competitions = ['best'], columns = ['hand','age','ht', 'rank']))
    atp_year_range.append(year)
  except:
    pass
all_coefs = np.array(all_coefs) #ATP

all_coefs2 = []
for year in year_range:
  try:
    all_coefs2.append(allfunc(years=[year], competitions = ['challengers'], columns = ['hand','age','ht', 'rank']))
    chall_year_range.append(year)
  except:
    pass
all_coefs2 = np.array(all_coefs2) #CHALLENGERS

all_coefs3 = []
for year in year_range:
  try:
    all_coefs3.append(allfunc(years=[year], competitions = ['futures'],columns = ['hand','age','ht', 'rank']))
    fut_year_range.append(year)
  except:
    pass
all_coefs3 = np.array(all_coefs3) #FUTURES

a = pd.read_csv("atp_matches_1977.csv")
a

n = np.array(year_range)
x_new = year_range
x_new

"""Use to check coefs"""

all_coefs[:,1]



"""### #1 Handedness"""

#no smooth
def graphing(x,y1,y2,y3):
  plt.ylim(-1, 1)
  plt.plot(x,y1, 'r', label="ATP")
  plt.plot(x,y2, 'g', label="Challengers")
  plt.plot(x,y3, 'b', label="Futures")
  plt.legend(loc="upper left")
  plt.axhline(y=0, color='k', linestyle=':')
  plt.xlabel('Time')
  plt.ylabel('Coefficients of Handedness')

graphing(x=atp_year_range, y1=all_coefs[:,0], y2=all_coefs2[:,0], y3=all_coefs3[:,0])

plt.clf()

#smooth
plt.ylim(-1.1, 1.1)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Handedness')

#a_BSpline = make_interp_spline(n, all_coefs[:,0], k=3)
#y_new = a_BSpline(x_new)

plt.rcParams["figure.figsize"] = (10, 10)
atp_year_range, all_coefs[:,0] = zip(*sorted(zip(atp_year_range, all_coefs[:,0])))
y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,0], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r',label='ATP' )

#a_BSpline = make_interp_spline(n, all_coefs2[:,0], k=3)
plt.rcParams["figure.figsize"] = (10, 10)
y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,0], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label = 'Challengers')

#a_BSpline = make_interp_spline(n, all_coefs3[:,0], k=3)
plt.rcParams["figure.figsize"] = (10, 10)
y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,0], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label = 'Futures')

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[atp, chall, fut])

plt.show()

lowess = sm.nonparametric.lowess
xs = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)
ys = np.sin(xs) + stats.cauchy.rvs(size=len(xs))
xs, ys = zip(*sorted(zip(xs, ys)))
z = lowess(ys, xs, frac= 1./3, it=0)
w = lowess(ys, xs, frac=1./3)
plt.plot(xs,ys)

plt.show()

plt.plot(xs,z)

"""### #2 Age"""

#no smooth
def graphing(x,y1,y2,y3):
  plt.ylim(-3, 3)
  plt.plot(x,y1, 'r', label="ATP")
  plt.plot(x,y2, 'g', label="Challengers")
  plt.plot(x,y3, 'b', label="Futures")
  plt.legend(loc="upper right")
  plt.axhline(y=0, color='k', linestyle=':')
  plt.xlabel('Time')
  plt.ylabel('Coefficients of Age')

graphing(x=year_range, y1=all_coefs[:,1], y2=all_coefs2[:,1], y3=all_coefs3[:,1])

#smooth
plt.ylim(-3, 3)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Age')

y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,1], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r', label="ATP")
plt.rcParams["figure.figsize"] = (10, 10)

y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,1], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label="Challengers")
plt.rcParams["figure.figsize"] = (10, 10)

y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,1], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label="Futures")
plt.rcParams["figure.figsize"] = (10, 10)

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[atp, chall, fut])
plt.show()

"""### #3 Height"""

#smooth
def graphing(x,y1,y2,y3):
  plt.ylim(-80, 80)
  plt.plot(x,y1, 'r', label="ATP")
  plt.plot(x,y2, 'g', label="Challengers")
  plt.plot(x,y3, 'b', label="Futures")
  plt.legend(loc="upper left")
  plt.axhline(y=0, color='k', linestyle=':')
  plt.xlabel('Time')
  plt.ylabel('Coefficients of Height')

graphing(x=year_range, y1=all_coefs[:,2], y2=all_coefs2[:,2], y3=all_coefs3[:,2])

#smooth
plt.ylim(-20, 20)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Height')

y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,2], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r', label="ATP")
plt.rcParams["figure.figsize"] = (10, 10)

y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,2], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label="Challengers")
plt.rcParams["figure.figsize"] = (10, 10)

y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,2], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label="Futures")
plt.rcParams["figure.figsize"] = (10, 10)

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[atp,chall,fut])
plt.show()

"""### #4 Rank"""

#no smooth
def graphing(x,y1,y2,y3):
  plt.ylim(-35, 35)
  plt.plot(x,y1, 'r', label="ATP")
  plt.plot(x,y2, 'g', label="Challengers")
  plt.plot(x,y3, 'b', label="Futures")
  plt.legend(loc="upper right")
  plt.axhline(y=0, color='k', linestyle=':')
  plt.xlabel('Time')
  plt.ylabel('Coefficients of Rank')

graphing(x=year_range, y1=all_coefs[:,3], y2=all_coefs2[:,3], y3=all_coefs3[:,3])

#smooth
plt.ylim(-35, 35)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Rank')

y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,3], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r', label="ATP")
plt.rcParams["figure.figsize"] = (10, 10)

y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,3], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label="Challengers")
plt.rcParams["figure.figsize"] = (10, 10)

y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,3], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label="Futures")
plt.rcParams["figure.figsize"] = (10, 10)

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles = [atp, chall, fut])
plt.show()

"""# Right-handed tennis players VS Left_handed tennis player

### ATP
"""

def preprocess(years=list(range(1968,2021)),competitions=["futures","challengers","best"]):
    full_df = None
    title ={"best":"",
            "challengers":"_qual_chall",
            "futures":"_futures"}
    for year in years:
      for comp in competitions:
        file_name = "atp_matches"+title[comp]+"_"+str(year)+".csv"
        if full_df is None:
          full_df = pd.read_csv(file_name)
        else:
          full_df = pd.concat([full_df,pd.read_csv(file_name)])
    return full_df


data = preprocess([
1991,
1992,
1993,
1994,
1995,
1996,
1997,
1998,
1999,
2000,
2001,
2002,
2003,
2004,
2005,
2006,
2007,
2008,
2009,
2010,
2011,
2012,
2013,
2014,
2015,
2016,
2017,
2018,
2019,
2020],['futures'])

data.shape

data = data[['winner_hand', 'loser_hand']]

data

data['winner_hand'].value_counts()

data['loser_hand'].value_counts()

24779+23438

data = data[(data['winner_hand']=="R") & (data['loser_hand']=="L") | (data['winner_hand']=="L") & (data['loser_hand']=="R")]

data['winner_hand'].value_counts()

data['loser_hand'].value_counts()

data

(24779/ 48217)*100

24779+  23438

df = pd.read_csv('atp_matches_qual_chall_2020.csv')

df.head()

for year in range(1968,2021):
  print(year)

"""# LHA without Nadal

"""

def preprocess(years=list(range(1981,2021)),competitions=["futures","challengers","best"]):
  full_df = None
  title ={"best":"",
          "challengers":"_qual_chall",
          "futures":"_futures"}
  for year in years:
    for comp in competitions:
      file_name = "atp_matches"+title[comp]+"_"+str(year)+".csv"
      if full_df is None:
        full_df = pd.read_csv(file_name)
      else:
        full_df = pd.concat([full_df,pd.read_csv(file_name)])
  return full_df

def cleaning(dataset, columns):
  if "hand" not in columns:
    columns.append("hand")
  dataset = dataset.loc[(dataset['winner_name']!='Rafael Nadal') & (dataset['loser_name']!='Rafael Nadal')]
  dataset = dataset.loc[(dataset['winner_hand']!='U') & (dataset['loser_hand']!='U')]
  all_cols = ["winner_" + col for col in columns] + ["loser_" + col for col in columns]
  data = dataset[all_cols]
  data = data[(data['winner_hand']=="R") & (data['loser_hand']=="L") | (data['winner_hand']=="L") & (data['loser_hand']=="R")]

  # fill the NaN values
  data = data.fillna(data.mean())
  data = pd.get_dummies(data, drop_first=True)
  data['winner_hand'] = data['winner_hand_R'].astype(float) # convert uint8 to float
  data['loser_hand'] = data['loser_hand_R'].astype(float)   # convert uint8 to float
  for col in columns:
    data[(col+'_dif')] = data[('winner_'+col)] - data[('loser_'+col)]

  flips = np.random.randint(0, 2, size=data.shape[0])
  data['winner'] = flips

  for col in columns:
    # xW + (W-1)*x
    data[(col+'_dif_rand')] = data[(col+'_dif')]* data['winner'] + (data['winner'] -1 )* data[(col+'_dif')]
  dif_cols = [col+"_dif_rand" for col in columns] +['winner']
  model_data = data[dif_cols]

  #normalizing
  x = model_data.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  model_data = pd.DataFrame(x_scaled)
  model_data.columns = dif_cols
  return model_data, data


def logreg(model_data):
  X_train, X_test, y_train, y_test = train_test_split(model_data.drop('winner', axis=1), model_data['winner'])
  model_logreg = LogisticRegression(penalty = 'none')
  model_logreg.fit(X_train, y_train)

  prediction = model_logreg.score(X_test, y_test)
  coeffs = model_logreg.coef_
  return coeffs.tolist()[0]

#takes in all previous functions
def allfunc(years=list(range(1968,2021)),competitions=["futures","challengers","best"], columns = ['hand','age','ht', 'rank', 'best_of']):
  data = preprocess(years, competitions)
  model_data, full_data = cleaning(data, columns)
  return logreg(model_data)

all_coefs = []
atp_year_range = []
chall_year_range = []
fut_year_range = []

year_range = list(range(1968,2021))
for year in year_range:
  try:
    all_coefs.append(allfunc(years=[year], competitions = ['best'],columns = ['hand','age','ht','rank']))
    atp_year_range.append(year)
  except:
    pass
all_coefs = np.array(all_coefs) #ATP

all_coefs2 = []
for year in year_range:
  try:
    all_coefs2.append(allfunc(years=[year], competitions = ['challengers'],columns = ['hand','age','ht','rank']))
    chall_year_range.append(year)
  except:
    pass
all_coefs2 = np.array(all_coefs2) #CHALLENGERS

all_coefs3 = []
for year in year_range:
  try:
    all_coefs3.append(allfunc(years=[year], competitions = ['futures'],columns = ['hand','age','ht','rank']))
    fut_year_range.append(year)
  except:
    pass
all_coefs3 = np.array(all_coefs3) #FUTURES

#smooth
plt.ylim(-1.1, 1.1)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Handedness')

#a_BSpline = make_interp_spline(n, all_coefs[:,0], k=3)
#y_new = a_BSpline(x_new)

plt.rcParams["figure.figsize"] = (10, 10)
atp_year_range, all_coefs[:,0] = zip(*sorted(zip(atp_year_range, all_coefs[:,0])))
y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,0], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r',label='ATP' )

#a_BSpline = make_interp_spline(n, all_coefs2[:,0], k=3)
plt.rcParams["figure.figsize"] = (10, 10)
y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,0], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label = 'Challengers')

#a_BSpline = make_interp_spline(n, all_coefs3[:,0], k=3)
plt.rcParams["figure.figsize"] = (10, 10)
y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,0], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label = 'Futures')

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[atp, chall, fut])

plt.show()

"""# Age advantage - (Big 3)"""

def preprocess(years=list(range(1981,2021)),competitions=["futures","challengers","best"]):
  full_df = None
  title ={"best":"",
          "challengers":"_qual_chall",
          "futures":"_futures"}
  for year in years:
    for comp in competitions:
      file_name = "atp_matches"+title[comp]+"_"+str(year)+".csv"
      if full_df is None:
        full_df = pd.read_csv(file_name)
      else:
        full_df = pd.concat([full_df,pd.read_csv(file_name)])
  return full_df

def cleaning(dataset, columns):
  if "hand" not in columns:
    columns.append("hand")
  dataset = dataset.loc[(dataset['winner_name']!='Rafael Nadal') & (dataset['loser_name']!='Rafael Nadal')]
  dataset = dataset.loc[(dataset['winner_name']!='Novak Djokovic') & (dataset['loser_name']!='Novak Djokovic')]
  dataset = dataset.loc[(dataset['winner_name']!='Roger Federer') & (dataset['loser_name']!='Roger Federer')]
  dataset = dataset.loc[(dataset['winner_hand']!='U') & (dataset['loser_hand']!='U')]
  all_cols = ["winner_" + col for col in columns] + ["loser_" + col for col in columns]
  data = dataset[all_cols]
  data = data[(data['winner_hand']=="R") & (data['loser_hand']=="L") | (data['winner_hand']=="L") & (data['loser_hand']=="R")]

  # fill the NaN values
  data = data.fillna(data.mean())
  data = pd.get_dummies(data, drop_first=True)
  data['winner_hand'] = data['winner_hand_R'].astype(float) # convert uint8 to float
  data['loser_hand'] = data['loser_hand_R'].astype(float)   # convert uint8 to float
  for col in columns:
    data[(col+'_dif')] = data[('winner_'+col)] - data[('loser_'+col)]

  flips = np.random.randint(0, 2, size=data.shape[0])
  data['winner'] = flips

  for col in columns:
    # xW + (W-1)*x
    data[(col+'_dif_rand')] = data[(col+'_dif')]* data['winner'] + (data['winner'] -1 )* data[(col+'_dif')]
  dif_cols = [col+"_dif_rand" for col in columns] +['winner']
  model_data = data[dif_cols]

  #normalizing
  x = model_data.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  model_data = pd.DataFrame(x_scaled)
  model_data.columns = dif_cols
  return model_data, data


def logreg(model_data):
  X_train, X_test, y_train, y_test = train_test_split(model_data.drop('winner', axis=1), model_data['winner'])
  model_logreg = LogisticRegression(penalty = 'none')
  model_logreg.fit(X_train, y_train)

  prediction = model_logreg.score(X_test, y_test)
  coeffs = model_logreg.coef_
  return coeffs.tolist()[0]

#takes in all previous functions
def allfunc(years=list(range(1968,2021)),competitions=["futures","challengers","best"], columns = ['hand','age','ht', 'rank', 'best_of']):
  data = preprocess(years, competitions)
  model_data, full_data = cleaning(data, columns)
  return logreg(model_data)

all_coefs = []
atp_year_range = []
chall_year_range = []
fut_year_range = []

year_range = list(range(1968,2021))
for year in year_range:
  try:
    all_coefs.append(allfunc(years=[year], competitions = ['best'],columns = ['hand','age','ht','rank']))
    atp_year_range.append(year)
  except:
    pass
all_coefs = np.array(all_coefs) #ATP

all_coefs2 = []
for year in year_range:
  try:
    all_coefs2.append(allfunc(years=[year], competitions = ['challengers'],columns = ['hand','age','ht','rank']))
    chall_year_range.append(year)
  except:
    pass
all_coefs2 = np.array(all_coefs2) #CHALLENGERS

all_coefs3 = []
for year in year_range:
  try:
    all_coefs3.append(allfunc(years=[year], competitions = ['futures'],columns = ['hand','age','ht','rank']))
    fut_year_range.append(year)
  except:
    pass
all_coefs3 = np.array(all_coefs3) #FUTURES

#smooth
plt.ylim(-3, 3)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Coefficients of Age')

y_range = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs[:,1], atp_year_range, frac=1./3)
atp,_ = plt.plot(atp_year_range, y_range, 'r', label="ATP")
plt.rcParams["figure.figsize"] = (10, 10)

y_range2 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs2[:,1], chall_year_range, frac =1./3)
chall,_ = plt.plot(chall_year_range, y_range2, 'g', label="Challengers")
plt.rcParams["figure.figsize"] = (10, 10)

y_range3 = statsmodels.nonparametric.smoothers_lowess.lowess(all_coefs3[:,1], fut_year_range, frac =1./3)
fut,_ = plt.plot(fut_year_range, y_range3, 'b', label="Futures")
plt.rcParams["figure.figsize"] = (10, 10)

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[atp, chall, fut])
plt.show()

