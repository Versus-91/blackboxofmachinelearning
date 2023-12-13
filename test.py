import numpy as np
import matplotlib.pyplot as plt

# Generating x values
x = np.linspace(-2, 2, 1000)

# Calculating y values for the function y = x + sin(x)
y = x + np.sin(3*x)

# Plotting the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = x + sin(x)', color='blue')
plt.plot(x, x, label='y = x + sin(x)', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x + sin(x) for x in (-4, 4)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()
