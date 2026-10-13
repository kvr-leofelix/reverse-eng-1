import numpy as np
import matplotlib.pyplot as plt
import os
# 1. Generate arbitrary 4D data
np.random.seed(42)
x = np.random.standard_normal(200)   # 1st Dimension (X axis)
y = np.random.standard_normal(200)   # 2nd Dimension (Y axis)
z = np.random.standard_normal(200)   # 3rd Dimension (Z axis)
w = np.random.standard_normal(200)   # 4th Dimension (Mapped to Color)

# 2. Set up the 3D plotting environment
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3. Create the scatter plot using the 'c' argument for the 4th dimension
img = ax.scatter(x, y, z, c=w, cmap='viridis', s=60, alpha=0.8, edgecolors='w')

# 4. Add labels and a colorbar to read the 4th dimension values
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
fig.colorbar(img, ax=ax, label='4th Dimension Value (W)')

plt.title('4D Scatter Plot (X, Y, Z + Color)')
plt.savefig('4d_plot.png', dpi=300, bbox_inches='tight')
print("Plot successfully saved to file.")


# Save the plot
plt.savefig('4d_plot.png', dpi=300, bbox_inches='tight')

# Print out exactly where it went
print(f"Graph saved securely at: {os.path.abspath('4d_plot.png')}")
