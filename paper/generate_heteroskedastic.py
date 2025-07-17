import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

from pysr import PySRRegressor

SEED = 0
N_ITERS = 900

QUANTILE = float(sys.argv[1])

# Parameters for the normal distribution
mean = 0
sample_size = 500
# scalar = 30000
scalar = 1

np.random.seed(SEED)
# Generate a sample from the univariate normal distribution
x_axis = np.arange(sample_size)
variances = x_axis**1.8
std_devs = np.sqrt(variances)  # Standard deviations for the distribution
relation = scalar* 3 *x_axis
data_raw = (relation + np.random.normal(loc=mean, scale=std_devs, size=sample_size))
data = data_raw / scalar

# Generate random x values for the scatter plot
x_values = np.arange(1, sample_size + 1)
z_01 = norm.ppf(0.01)  # z-score for the 50th quantile (median)
z_50 = norm.ppf(0.5)  # z-score for the 50th quantile (median)
z_75 = norm.ppf(0.75) # z-score for the 75th quantile
z_90 = norm.ppf(0.99)  # z-score for the 90th quantile
quantiles_01 = (relation + ( mean + std_devs * z_01))/scalar  # Conditional medians (50th quantiles)
quantiles_50 = (relation + ( mean + std_devs * z_50))/scalar  # Conditional medians (50th quantiles)
quantiles_75  = (relation + ( mean + std_devs * z_75))/scalar  # Conditional medians (50th quantiles)
quantiles_90 = (relation + (mean + std_devs * z_90 ))/scalar # Conditional 90th quantiles

print("% Scatter")
print("\\addplot[mark=*, fill=gray, opacity=0.3, only marks] coordinates {")
for x in x_axis:
    print("  ({}, {})".format(x, data[x]))
print("};")


print("% Q01 line plot")
print("\\addplot[color=red, dotted, line width=1.2] plot[smooth] coordinates {")
for x in x_axis:
    print("  ({},{})".format(x, quantiles_01[x]))
print("};")
print("\\node[anchor=west, color=red] at (axis cs:{}, {}) {{$\\tau=0.01$}}; % Text to the right of the last point".format(x_axis[-1], quantiles_01[-1]))


print("% Q50 line plot")
print("\\addplot[color=blue,line width=1.2] plot[smooth] coordinates {")
for x in x_axis:
    print("  ({},{})".format(x, quantiles_50[x]))
print("};")
print("\\node[anchor=west, color=blue] at (axis cs:{}, {}) {{$\\tau=0.5$}}; % Text to the right of the last point".format(x_axis[-1], quantiles_50[-1]))


print("% Q75 line plot")
print("\\addplot[color=teal, dash dot,line width=2] plot[smooth] coordinates {")
for x in x_axis:
    print("  ({},{})".format(x, quantiles_75[x]))
print("};")
print("\\node[anchor=west, color=teal] at (axis cs:{}, {}) {{$\\tau=0.75$}}; % Text to the right of the last point".format(x_axis[-1], quantiles_75[-1]))


print("% Q90 line plot")
print("\\addplot[color=orange, dash dot dot,line width=2] plot[smooth] coordinates {")
for x in x_axis:
    print("  ({},{})".format(x, quantiles_90[x]))
print("};")
print("\\node[anchor=west, color=orange] at (axis cs:{}, {}) {{$\\tau=0.99$}}; % Text to the right of the last point".format(x_axis[-1], quantiles_90[-1]))



# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, data, alpha=0.7, color='grey', edgecolor='k')
plt.plot(quantiles_50, color='red')
plt.plot(quantiles_90, color='blue')
plt.title('Scatter Plot of a Sample from a Univariate Normal Distribution')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Complexity parameters
binary_operators = ["+", "*", "-", "/"]
unary_operators = ["exp",]
complexity_of_operators = {
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 1,
    "exp": 2,
}


params_sqr = {
    "niterations": N_ITERS,  # improve for better results50
    "binary_operators": binary_operators,
    "unary_operators": unary_operators,
    "complexity_of_operators": complexity_of_operators,
    "elementwise_loss": f"QuantileLoss({QUANTILE})",
    "deterministic": True,
    "parallelism": "serial",
    "temp_equation_file": True,
    "parsimony": 0.0,
    "progress": True,
    # "verbosity": 0,
    "batch_size": 5000,
    "random_state": SEED,
}

data_sqr = data

modelq = PySRRegressor(
    **params_sqr
)
modelq.fit(x_axis.reshape(-1,1), data_sqr.reshape(-1,1))

print("======", QUANTILE, "======")
print(modelq)
print('test')