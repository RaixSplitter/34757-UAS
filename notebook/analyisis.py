from turtle import color
import pandas as pd
import matplotlib.pyplot as plt

FILEPATH = "data/out.csv"
FREQUENCY = 120 #Hz
START = 0 # seconds
END = -1 # seconds

df = pd.read_csv(FILEPATH, sep=",", header=None, names=["Thrust", "X", "Y", "Z", "QX", "QY", "QZ", "QW"])
df = df.iloc[FREQUENCY*START:FREQUENCY*END].reset_index(drop=True)

checkpoint = df.iloc[::FREQUENCY].reset_index(drop=True)

fig = plt.figure(figsize=(8, 6), layout='constrained')
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['X'], df['Y'], df['Z'], marker='o', linestyle='-', markersize=2)
ax.scatter(df['X'][0], df['Y'][0], df['Z'][0], c='r', marker='o', s=10, label='Data Points')

ax.scatter(checkpoint['X'], checkpoint['Y'], checkpoint['Z'], c='g', marker='o', s=20, label='Checkpoint Points')

# Add visible index labels to the checkpoints
for idx, (x, y, z) in checkpoint[['X', 'Y', 'Z']].iterrows():
    ax.text(x, y, z, str(idx), color='black', fontsize=8, ha='center', va='bottom')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of XYZ Coordinates')
plt.show()