import matplotlib.pyplot as plt
from scipy import stats

# Larger dataset
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [4, 3, 6, 8.5, 7, 11.8, 10.2, 11.4, 19.1, 20.2]

# Perform linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Define regression function
def predict(x):
    return slope * x + intercept

# Predicted values for regression line
line = [predict(i) for i in x]

# Print regression results
print(f"slope = {slope:.3f}, intercept = {intercept:.3f}, r = {r:.3f}, p = {p:.3e}, std_err = {std_err:.3f}")

# Plot original data
plt.scatter(x, y, color="blue", label="Data points")

# Plot regression line
plt.plot(x, line, color="red", label="Regression line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()