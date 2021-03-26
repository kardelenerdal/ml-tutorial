import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("data.csv")

# set for checking duplicates
theSet = set([])
# removing duplicates
for x in dataframe["Theta"]:
    if theSet.__contains__(x):
        indexNames = dataframe[dataframe['Theta'] == x].index
        dataframe.drop(indexNames[1], inplace=True)
    else:
        theSet.add(x)

df = dataframe.sort_values(by='Theta')
f1 = plt.figure(figsize=(30, 15))

ax1 = f1.add_subplot(3, 3, 1)
ax1.plot(df["Theta"], df["A-X (meter)"])
ax1.set_ylabel('A-X (meter)')
ax1.set_xlabel('Theta')
ax1.grid(True)
ax1.set_xlim([-3.14, 3.14])

ax2 = f1.add_subplot(3, 3, 4)
ax2.plot(df["Theta"], df["A-Y (meter)"])
ax2.set_ylabel('A-Y (meter)')
ax2.set_xlabel('Theta')
ax2.grid(True)
ax2.set_xlim([-3.14, 3.14])

ax3 = f1.add_subplot(3, 3, 2)
ax3.plot(df["Theta"], df["A-X Vel (meter/sec)"])
ax3.set_ylabel('A-X Vel (meter/sec)')
ax3.set_xlabel('Theta')
ax3.grid(True)
ax3.set_xlim([-3.14, 3.14])

ax4 = f1.add_subplot(3, 3, 3)
ax4.plot(df["Theta"], df["A-X Acc (meter/sec**2)"])
ax4.set_ylabel('A-X Acc (meter/sec**2)')
ax4.set_xlabel('Theta')
ax4.grid(True)
ax4.set_xlim([-3.14, 3.14])

ax5 = f1.add_subplot(3, 3, 5)
ax5.plot(df["Theta"], df["A-Y Vel (meter/sec)"])
ax5.set_ylabel('A-Y Vel (meter/sec)')
ax5.set_xlabel('Theta')
ax5.grid(True)
ax5.set_xlim([-3.14, 3.14])

ax6 = f1.add_subplot(3, 3, 6)
ax6.plot(df["Theta"], df["A-Y Acc (meter/sec**2)"])
ax6.set_ylabel('A-Y Acc (meter/sec**2)')
ax6.set_xlabel('Theta')
ax6.grid(True)
ax6.set_xlim([-3.14, 3.14])

"""ax7 = f1.add_subplot(3, 3, 7)
ax7.plot(df["Theta"], df["A-Y Acc (meter/sec**2)"])
ax7.set_ylabel('A-Y Acc (meter/sec**2)')
ax7.set_xlabel('Theta')
ax7.grid(True)
ax7.set_xlim([-3.14, 3.14])"""

plt.suptitle('Kinematic Analysis For Point A')
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.show()

