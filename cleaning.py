import matplotlib.pyplot as plt
import datetime as dt
import missingno as msno
import os
import seaborn as sns
import numpy as np
import pandas as pd
os.getcwd()
data1=pd.read_csv(r'C:\Users\Ismat\Downloads\Housing.csv')
data1
data1.info()
data1.head()
data1.tail()
len(data1)
data1.columns
data1.describe()
data1.describe().transpose()
data1.sum()
data1.mean()
data1.count()
data1.shape
data1.columns=['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'main_road',
       'guestroom', 'basement', 'hot_water_heating', 'air_conditioning',
       'parking', 'pref_area', 'furnishing_status']
data1['avg_price']=data1['price']*1000/len(data1)
data1=data1.rename(columns = {'price' : 'Cost'}).head(2)
data1['avg_price']=round(data1['avg_price'].astype(int))
data1.head()
data1['guestroom'].value_counts(dropna=False) 
data1['guestroom'].nunique()
data1.drop('avg_price',axis=1)
data2['Gender'].np.where()
data1.groupby(['Gender']).agg({'ApplicantIncome':'mean'})
data1['Price']=pd.to_numeric(data1['Price'],errors='coerce')
data1['Date']=pd.to_datetime(data1['Date'])
data1['Weekday'] = data1['Date'].dt.weekday + 1
data1['Weekday']['Price'].mean()
data2=data1.groupby(['Weekday']).mean('Price')  # Step 3
data1['Avg_per_week_day']=data1.groupby('Weekday')[['Price']].mean()
data4=data1.groupby('Weekday')[['Price']].mean()
data4['s/s']=data4.index
plt.hist(data2['Price'])
plt.show()

x = np.random.normal(170, 10, 250)

x=data2['Price']
y=data2['s/s']
sns.set_style('darkgrid')
sns.distplot(x,bins=y)

data2['s/s']=data2.index
data2.corr()

x=data2.corr()
sns.heatmap(x,xticklabels=x.columns.values,yticklabels=x.columns.values);


data2.groupby(['Price']).std()

data2['Price'].value_counts()

sns.countplot(data2['Price'])

sns.scatterplot(x='s/s',y='Price',data=data2)

data2.corr()['Price']['s/s']

sns.jointplot(x='s/s',y='Price',data=data2)

sns.lmplot(x='s/s',y='Price',data=data2)

sns.catplot (x='s/s',y='Price',data=data2)

sns.violinplot(y='Price',data=data2)

data1.isna().sum()

missingno.matrix(data1,figsize=(20,10))


# Using numpy random function to generate random data
np.random.seed(19685689) 
mu, sigma = 120, 30
x = mu + sigma * np.random.randn(10000)
# passing the histogram function
n, bins, patches = plt.hist(x, 70, histtype='bar', density=True, facecolor='red', alpha=0.99)
plt.xlabel('Values')
plt.ylabel('Probability Distribution')
plt.title('Histogram showing Data Distribution')
plt.xlim(50, 180)
plt.ylim(0, 0.04)
plt.grid(True)
plt.show()


from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
# Forming the dataset with numpy random function
np.random.seed(190345678) 
N_points = 100000
n_bins = 40
# Creating distribution 
x = np.random.randn(N_points) 
y = .10 ** x + np.random.randn(100000) + 25
legend = ['distribution'] 
# Passing subplot function
fig, axs = plt.subplots(1, 1, figsize =(10, 7),  tight_layout = True) 
# Removing axes spines  
for s in ['top', 'bottom', 'left', 'right']:  
    axs.spines[s].set_visible(False)  
# Removing x, y ticks 
axs.xaxis.set_ticks_position('none')  
axs.yaxis.set_ticks_position('none')  
# Adding padding between axes and labels  
axs.xaxis.set_tick_params(pad = 7)  
axs.yaxis.set_tick_params(pad = 15)  
# Adding x, y gridlines  
axs.grid(b = True, color ='pink',  linestyle ='-.', linewidth = 0.6,  alpha = 0.6)  
# Passing histogram function
N, bins, patches = axs.hist(x, bins = n_bins) 
# Setting the color 
fracs = ((N**(1 / 5)) / N.max()) 
norm = colors.Normalize(fracs.min(), fracs.max()) 
for thisfrac, thispatch in zip(fracs, patches): 
    color = plt.cm.viridis_r(norm(thisfrac)) 
    thispatch.set_facecolor(color) 
# Adding extra features for making it more presentable    
plt.xlabel("X-axis") 
plt.ylabel("y-axis") 
plt.legend(legend) 
plt.title('Customizing your own histogram') 
plt.show()



np.random.seed(9**7) 
n_bins = 15
x = np.random.randn(10000, 5)   
colors = ['blue', 'pink', 'orange','green','red'] 
plt.hist(x, n_bins, density = True,  histtype ='step', color = colors, label = colors) 
plt.legend(prop ={'size': 10}) 
plt.show()



np.random.seed(9**7) 
n_bins = 15
x = np.random.randn(10000, 5) 
colors = ['blue', 'pink', 'orange','green','red'] 
plt.hist(x, n_bins, density = True,  histtype ='bar', color = colors, label = colors) 
plt.legend(prop ={'size': 10}) 
plt.show()


# Generating random data
n = 1000
x = np.random.standard_normal(1000)
y = 5.0 * x + 3.0* np.random.standard_normal(1000)  
fig = plt.subplots(figsize =(10, 7))
# Plotting 2D Histogram
plt.hist2d(x, y,bins=50)
plt.title("2D Histogram")
plt.show()



# Creating dataset
a = np.array([22, 87, 5, 43, 56,
              73, 55, 54, 11,
              20, 51, 5, 79, 31,
              27]) 
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = [0, 25, 50, 75, 100])
# Show plot
plt.show()




msno.bar(data)
data.notna().sum() / data.shape[0] * 100
data.fillna('No info')
data.loc[data.index.isin(range(int(len(data)/2))), 'A'] = 1
# for finding null values
msno.matrix(data)
data.isna().sum()
data.isna().any()
data.isna().any()[data.isna().any()]
data.isna().sum() / len(data) *100
data.notna().sum() / len(data) * 100
# fillna (4 default type) - str, int, float, bool
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data.dropna(axis = 'columns', thresh = int(len(data)*0.95))
msno.bar(data)
columns_not_na = data.isna().any()[data.isna().any()].index
data.fillna(data.mean())
data.dtypes
#for columns null values percentage 
100 - data.isna().sum() / len(data) * 100
#### Weekday with number
data['Weekday'] = data['Date'].dt.weekday
#### Weekday with name
data['Weekday_name'] = data['Date'].dt.weekday_name
data['Year'] = data['Date'].dt.year
#### Day of year
data['Day_of_Year'] = data['Date'].dt.dayofyear
#### Month with number
data['Month'] = data['Date'].dt.month
#### Month with name
data['Month_name'] = data['Date'].dt.month_name()
#### Difference from today
data['Day_difference'] = pd.datetime.today() - data['Date']

#### Convert day difference to month difference with np.timedelta64

    * 'M' - for month difference
    * 'Y' - for year difference    
data['Month_difference'] = data['Day_difference']/np.timedelta64(1,'M')
data['Year_difference'] = data['Day_difference']/np.timedelta64(1,'Y')




