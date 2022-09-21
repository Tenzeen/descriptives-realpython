import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#calculating descriptive statistics
x = [8.0, 1, 2.5, 4, 28.0] #create a list of data
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0] #create a list of data containing nan value
#print
x 
x_with_nan

#create np.ndarray and pd.Series corresponding to x and x_with_nan
y, y_with_nan = np.array(x), np.array(x_with_nan) #numpy array
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan) #pandas series

#print numpy array
y
y_with_nan
#print pandas series
z
z_with_nan

#measures of central tendency
#mean
mean_ = sum(x) / len(x) #pure python method for calculating mean
mean_
mean_ = statistics.mean(x) #method for calculating mean by applying built-in python statistics functions
mean_
mean_ = statistics.fmean(x) #always returns floating number and is faster than mean.
mean_

#if nan values are included in the data, statistics.mean() will return nan for the output
mean_ = statistics.mean(x_with_nan)
mean_
mean_ = statistics.fmean(x_with_nan)
mean_

#find mean using numpy
mean_ = np.mean(y)
mean_
mean_ = y.mean() #can also use .mean()
mean_

#The function mean() and method .mean() from NumPy return the same result as statistics.mean(). This is also the case when there are nan values among your data
np.mean(y_with_nan)
y_with_nan.mean()

#If you prefer to ignore nan values, then you can use:
np.nanmean(y_with_nan)

#pandas series for calculating mean
mean_ = z.mean()
mean_

#.mean() from Pandas ignores nan values by default
z_with_nan.mean()

#weighted mean
#You can implement the weighted mean in pure Python by combining sum() with either range() or zip()
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

#Numpy is better for larger datasets. You can use np.average() to get the weighted mean of NumPy arrays or Pandas Series
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean
wmean = np.average(z, weights=w)
wmean

#can also use product w * y with np.sum() or .sum() to calulate weighted mean
(w * y).sum() / w.sum()

#weighted mean for datasets with nan values
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()
np.average(y_with_nan, weights=w)
np.average(z_with_nan, weights=w)

#Harmonic mean
hmean = len(x) / sum(1 / item for item in x) #pure python method for calculating harmonic mean
hmean

#can also calculate this measure with statistics.harmonic_mean()
hmean = statistics.harmonic_mean(x)
hmean
statistics.harmonic_mean(x_with_nan) #returns nan if there is a nan value
statistics.harmonic_mean([1, 0, 2]) #returns 0 if theres a 0 value
statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError if there is a negative number

#another way to calculate the harmonic mean is to use scipy.stats.hmean()
scipy.stats.hmean(y)
scipy.stats.hmean(z)

#geometric mean
#pure python method for calculating geometric mean
gmean = 1
for item in x:
   gmean *= item
gmean **= 1 / len(x)
gmean

#can use python statistics to convert all values to floating-point numbers and returns their geometric mean
gmean = statistics.geometric_mean(x)
gmean

#returns nan if there are nan values
gmean = statistics.geometric_mean(x_with_nan)
gmean

#can also use scipy to calculate geometric mean
scipy.stats.gmean(y)
scipy.stats.gmean(z)

#Median
#pure python method for calculating median
n = len(x)
if n % 2:
     median_ = sorted(x)[round(0.5*(n-1))]
else:
     x_ord, index = sorted(x), round(0.5 * n)
     median_ = 0.5 * (x_ord[index-1] + x_ord[index])
median_

#can get median by using python statistics
median_ = statistics.median(x)
median_
median_ = statistics.median(x[:-1])
median_

#if there are 2 middle values
statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

#median(), median_low(), and median_high() don’t return nan when there are nan values
statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

#can also get median using numpy
median_ = np.median(y)
median_
median_ = np.median(y[:-1])
median_

#using nanmedian ignores nan values
np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

#Pandas Series objects have the method .median() that ignores nan values by default
z.median()
z_with_nan.median()

#Mode
#using pure python
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

#using python statistics
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_

#if there is more than one modal value:
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v) #returns list with all modes

#statistics.mode() and statistics.multimode() handle nan values as regular values and can return nan as the modal value
statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])
statistics.multimode([2, math.nan, 0, math.nan, 5])

#can also use scipy for mode
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
mode_ = scipy.stats.mode(v)
mode_

#You can get the mode and its number of occurrences as NumPy arrays with dot notation
mode_.mode
mode_.count

#Pandas Series objects have the method .mode() that handles multimodal values well and ignores nan values by default
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()

#Measures of Variability
#calculate sample variance with pure python
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

#calculate sample variance with python statistics
var_ = statistics.variance(x)
var_
#If you have nan values among your data, then statistics.variance() will return nan
statistics.variance(x_with_nan)

#You can also calculate the sample variance with NumPy. You should use the function np.var() or the corresponding method .var()
var_ = np.var(y, ddof=1)
var_
var_ = y.var(ddof=1)
var_
#If you have nan values in the dataset, then np.var() and .var() will return nan:
np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)
#If you want to skip nan values, then you should use np.nanvar()
np.nanvar(y_with_nan, ddof=1)

#pandas series objects have the method .var() that skips nan values by default
z.var(ddof=1)
z_with_nan.var(ddof=1)

#standard deviation
#calculating standard deviation using pure python
std_ = var_ ** 0.5
std_

#using python statistics
std_ = statistics.stdev(x)
std_

#using numpy
np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
np.nanstd(y_with_nan, ddof=1) #ignores nan values

#pandas series objects also have the method .std() that skips nan by default
z.std(ddof=1)
z_with_nan.std(ddof=1)

#skewness
#calculate using pure python
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_ #skew is positive so x as a right side tail

#You can also calculate the sample skewness with scipy
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False)

#percentiles
#calculate using python statistics
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x, n=4, method='inclusive')

#using numpy
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

# can also be a sequence of numbers:
np.percentile(y, [25, 50, 75])
np.median(y)

#If you want to ignore nan values, then use np.nanpercentile() instead
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

#using quantile and nanquantile in numpy with percentiles as a decimal
np.quantile(y, 0.05)
np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

#pandas series objects have the method .quantile()
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

#Ranges
#calculate using numpy
np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

#you can use built-in Python, NumPy, or Pandas functions and methods to calculate the maxima and minima of sequences
np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

#The interquartile range is the difference between the first and third quartile. Once you calculate the quartiles, you can take their difference
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]
quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

#summary of descriptive statistics
#get summary of data using scipy
result = scipy.stats.describe(y, ddof=1, bias=False)
result

#You can access particular values with dot notation
result.nobs
result.minmax[0]  # Min
result.minmax[1]  # Max
result.mean
result.variance
result.skewness
result.kurtosis

#pandas series objects have the method .describe()
result = z.describe()
result

#you can access each item of result with its label
result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

#Measures of Correlation Between Pairs of Data
#create two Python lists and use them to get corresponding NumPy arrays and Pandas Series
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#Covariance
#calculate the covariance in pure Python
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
           / (n - 1))
cov_xy

#using numpy
#The upper-left element of the covariance matrix is the covariance of x and x, or the variance of x. 
#Similarly, the lower-right element is the covariance of y and y, or the variance of y.
cov_matrix = np.cov(x_, y_)
cov_matrix

#Check to verify if true
x_.var(ddof=1)
y_.var(ddof=1)

#The other two elements of the covariance matrix are equal and represent the actual covariance between x and y
cov_xy = cov_matrix[0, 1]
cov_xy
cov_xy = cov_matrix[1, 0]
cov_xy

#Pandas Series have the method .cov() that you can use to calculate the covariance
cov_xy = x__.cov(y__)
cov_xy
cov_xy = y__.cov(x__)
cov_xy

#Correlation Coefficient
#using pure python
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

#scipy.stats has the routine pearsonr() that calculates the correlation coefficient and the 𝑝-value
r, p = scipy.stats.pearsonr(x_, y_)
r
p


#using numpy
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

#The other two elements are equal and represent the actual correlation coefficient between x_ and y_
r = corr_matrix[0, 1]
r
r = corr_matrix[1, 0]
r

#you can get the correlation coefficient with scipy.stats.linregress()
scipy.stats.linregress(x_, y_)

#To access particular values from the result of linregress(), including the correlation coefficient, use dot notation
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

#Pandas Series have the method .corr() for calculating the correlation coefficient
r = x__.corr(y__)
r
r = y__.corr(x__)
r

#Working with 2d Data
#Start by creating a 2D NumPy array:
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a

#You can apply Python statistics functions and methods to it just as you would to 1D data
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)

#axis=None says to calculate the statistics across all data in the array. The examples above work like this. This behavior is often the default in NumPy.
#axis=0 says to calculate the statistics across all rows, that is, for each column of the array. This behavior is often the default for SciPy statistical functions.
#axis=1 says to calculate the statistics across all columns, that is, for each row of the array.
np.mean(a, axis=0)
a.mean(axis=0)

#If you provide axis=1 to mean(), then you’ll get the results for each row:
np.mean(a, axis=1)
a.mean(axis=1)

#The parameter axis works the same way with other NumPy functions and methods:
np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

#This is very similar when you work with SciPy statistics functions. But remember that in this case, the default value for axis is 0
scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0)

#If you specify axis=1, then you’ll get the calculations across all columns, that is for each row
scipy.stats.gmean(a, axis=1)

#If you want statistics for the entire dataset, then you have to provide axis=None
scipy.stats.gmean(a, axis=None)

#You can get a Python statistics summary with a single function call for 2D data with scipy.stats.describe()
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)

#You can get a particular value from the summary with dot notation:
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

#DataFrames
#Use the array a and create a DataFrame
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

#If you call Python statistics methods without arguments, then the DataFrame will return the results for each column
df.mean()
df.var()

#If you want the results for each row, then just specify the parameter axis=1
df.mean(axis=1)
df.var(axis=1)

#You can isolate each column of a DataFrame like this
df['A']

#Now, you have the column 'A' in the form of a Series object and you can apply the appropriate methods
df['A'].mean()
df['A'].var()

#It’s possible to get all data from a DataFrame with .values or .to_numpy()
df.values
df.to_numpy()

#Like Series, DataFrame objects have the method .describe() that returns another DataFrame with the statistics summary for all columns
df.describe()

#You can access each item of the summary like this
df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

#visualizing data
#boxplots
#First, create some data to represent with a box plot
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

#Now that you have the data to work with, you can apply .boxplot() to get the box plot
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#Histograms
#The function np.histogram() is a convenient way to get data for histograms
hist, bin_edges = np.histogram(x, bins=10)
hist # contains the frequency or the number of items corresponding to each bin.
bin_edges #contains the edges or bounds of the bin.

#What histogram() calculates, .hist() can show graphically
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#It’s possible to get the histogram with the cumulative numbers of items if you provide the argument cumulative=True to .hist()
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#Pie charts
#define data associated to three labels:
x, y, z = 128, 256, 1024

#Now, create a pie chart with .pie():
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#bar charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

#You can create a bar chart with .bar() if you want vertical bars or .barh() if you’d like horizontal bars
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#X-Y Plots
#generate two datasets and perform linear regression with scipy.stats.linregress():
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

#Then you can apply .plot() to get the x-y plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#Heatmaps
#You can create the heatmap for a covariance matrix with .imshow()
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

#You can obtain the heatmap for the correlation coefficient matrix following the same logic
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()




















