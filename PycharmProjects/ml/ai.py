import math
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
        dataframe.drop(indexNames[0], inplace=True)
    else:
        theSet.add(x)

df = dataframe.sort_values(by='Theta')

mag_pos_a = []
mag_vel_a = []
mag_acc_a = []

mag_pos_b = []
mag_vel_b = []
mag_acc_b = []

mag_pos_c = []
mag_vel_c = []
mag_acc_c = []

mag_pos_d = []
mag_vel_d = []
mag_acc_d = []

mag_pos_d0 = []
mag_vel_d0 = []
mag_acc_d0 = []

for x, y in zip(df["A-X (meter)"], df["A-Y (meter)"]):
    mag_pos_a.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["A-X Vel (meter/sec)"], df["A-Y Vel (meter/sec)"]):
    mag_vel_a.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["A-X Acc (meter/sec**2)"], df["A-Y Acc (meter/sec**2)"]):
    mag_acc_a.append(math.sqrt((x*x)+(y*y)))

for x, y in zip(df["B-X (meter)"], df["B-Y (meter)"]):
    mag_pos_b.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["B-X Vel (meter/sec)"], df["B-Y Vel (meter/sec)"]):
    mag_vel_b.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["B-X Acc (meter/sec**2)"], df["B-Y Acc (meter/sec**2)"]):
    mag_acc_b.append(math.sqrt((x*x)+(y*y)))

for x, y in zip(df["C-X (meter)"], df["C-Y (meter)"]):
    mag_pos_c.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["C-X Vel (meter/sec)"], df["C-Y Vel (meter/sec)"]):
    mag_vel_c.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["C-X Acc (meter/sec**2)"], df["C-Y Acc (meter/sec**2)"]):
    mag_acc_c.append(math.sqrt((x*x)+(y*y)))

for x, y in zip(df["D-X (meter)"], df["D-Y (meter)"]):
    mag_pos_d.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["D-X Vel (meter/sec)"], df["D-Y Vel (meter/sec)"]):
    mag_vel_d.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["D-X Acc (meter/sec**2)"], df["D-Y Acc (meter/sec**2)"]):
    mag_acc_d.append(math.sqrt((x*x)+(y*y)))

for x, y in zip(df["D0-X (meter)"], df["D0-Y (meter)"]):
    mag_pos_d0.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["D0-X Vel (meter/sec)"], df["D0-Y Vel (meter/sec)"]):
    mag_vel_d0.append(math.sqrt((x*x)+(y*y)))
for x, y in zip(df["D0-X Acc (meter/sec**2)"], df["D0-Y Acc (meter/sec**2)"]):
    mag_acc_d0.append(math.sqrt((x*x)+(y*y)))

df["Magnitude Position A"] = mag_pos_a
df["Magnitude Velocity A"] = mag_vel_a
df["Magnitude Acc A"] = mag_acc_a

df["Magnitude Position B"] = mag_pos_b
df["Magnitude Velocity B"] = mag_vel_b
df["Magnitude Acc B"] = mag_acc_b

df["Magnitude Position C"] = mag_pos_c
df["Magnitude Velocity C"] = mag_vel_c
df["Magnitude Acc C"] = mag_acc_c

df["Magnitude Position D"] = mag_pos_d
df["Magnitude Velocity D"] = mag_vel_d
df["Magnitude Acc D"] = mag_acc_d

df["Magnitude Position D0"] = mag_pos_d0
df["Magnitude Velocity D0"] = mag_vel_d0
df["Magnitude Acc D0"] = mag_acc_d0

f1 = plt.figure(figsize=(15.2, 8.4))

ax1 = f1.add_subplot(3, 3, 1)
ax1.plot(df["Theta"], df["A-X (meter)"])
ax1.set_ylabel('A-X (meter)', fontsize=8)
ax1.set_xlabel('θ12', fontsize=8)
ax1.grid(True)
ax1.set_xlim([-3.14, 3.14])

ax2 = f1.add_subplot(3, 3, 4)
ax2.plot(df["Theta"], df["A-Y (meter)"])
ax2.set_ylabel('A-Y (meter)', fontsize=8)
ax2.set_xlabel('θ12', fontsize=8)
ax2.grid(True)
ax2.set_xlim([-3.14, 3.14])

ax3 = f1.add_subplot(3, 3, 2)
ax3.plot(df["Theta"], df["A-X Vel (meter/sec)"])
ax3.set_ylabel('A-X Vel (meter/sec)', fontsize=8)
ax3.set_xlabel('θ12', fontsize=8)
ax3.grid(True)
ax3.set_xlim([-3.14, 3.14])

ax4 = f1.add_subplot(3, 3, 3)
ax4.plot(df["Theta"], df["A-X Acc (meter/sec**2)"])
ax4.set_ylabel('A-X Acc (meter/sec**2)', fontsize=8)
ax4.set_xlabel('θ12', fontsize=8)
ax4.grid(True)
ax4.set_xlim([-3.14, 3.14])

ax5 = f1.add_subplot(3, 3, 5)
ax5.plot(df["Theta"], df["A-Y Vel (meter/sec)"])
ax5.set_ylabel('A-Y Vel (meter/sec)', fontsize=8)
ax5.set_xlabel('θ12', fontsize=8)
ax5.grid(True)
ax5.set_xlim([-3.14, 3.14])

ax6 = f1.add_subplot(3, 3, 6)
ax6.plot(df["Theta"], df["A-Y Acc (meter/sec**2)"])
ax6.set_ylabel('A-Y Acc (meter/sec**2)', fontsize=8)
ax6.set_xlabel('θ12', fontsize=8)
ax6.grid(True)
ax6.set_xlim([-3.14, 3.14])

ax7 = f1.add_subplot(3, 3, 7)
ax7.plot(df["Theta"], df["Magnitude Position A"])
ax7.set_ylabel('Magnitude Of Position Vector (m)', fontsize=8)
ax7.set_xlabel('θ12', fontsize=8)
ax7.grid(True)
ax7.set_ylim([-0.1, 0.1])

ax8 = f1.add_subplot(3, 3, 8)
ax8.plot(df["Theta"], df["Magnitude Velocity A"])
ax8.set_ylabel('Magnitude Of Velocity Vector (m/s)', fontsize=8)
ax8.set_xlabel('θ12', fontsize=8)
ax8.grid(True)
ax8.set_ylim([-0.1, 1.0])

ax9 = f1.add_subplot(3, 3, 9)
ax9.plot(df["Theta"], df["Magnitude Acc A"])
ax9.set_ylabel('Magnitude Of Acceleration Vector (m/s**2)', fontsize=8)
ax9.set_xlabel('θ12', fontsize=8)
ax9.grid(True)
ax9.set_ylim([-0.1, 1.0])

plt.suptitle('Kinematic Analysis For Point A', fontsize=20)
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.savefig('outputA.png')

####################################################
f2 = plt.figure(figsize=(15.2, 8.4))

ax1b = f2.add_subplot(3, 3, 1)
ax1b.plot(df["Theta"], df["B-X (meter)"])
ax1b.set_ylabel('B-X (meter)', fontsize=8)
ax1b.set_xlabel('θ12', fontsize=8)
ax1b.grid(True)
ax1b.set_xlim([-3.14, 3.14])

ax2b = f2.add_subplot(3, 3, 4)
ax2b.plot(df["Theta"], df["B-Y (meter)"])
ax2b.set_ylabel('B-Y (meter)', fontsize=8)
ax2b.set_xlabel('θ12', fontsize=8)
ax2b.grid(True)
ax2b.set_xlim([-3.14, 3.14])

ax3b = f2.add_subplot(3, 3, 2)
ax3b.plot(df["Theta"], df["B-X Vel (meter/sec)"])
ax3b.set_ylabel('B-X Vel (meter/sec)', fontsize=8)
ax3b.set_xlabel('θ12', fontsize=8)
ax3b.grid(True)
ax3b.set_xlim([-3.14, 3.14])

ax4b = f2.add_subplot(3, 3, 3)
ax4b.plot(df["Theta"], df["B-X Acc (meter/sec**2)"])
ax4b.set_ylabel('B-X Acc (meter/sec**2)', fontsize=8)
ax4b.set_xlabel('θ12', fontsize=8)
ax4b.grid(True)
ax4b.set_xlim([-3.14, 3.14])

ax5b = f2.add_subplot(3, 3, 5)
ax5b.plot(df["Theta"], df["B-Y Vel (meter/sec)"])
ax5b.set_ylabel('B-Y Vel (meter/sec)', fontsize=8)
ax5b.set_xlabel('θ12', fontsize=8)
ax5b.grid(True)
ax5b.set_xlim([-3.14, 3.14])

ax6b = f2.add_subplot(3, 3, 6)
ax6b.plot(df["Theta"], df["B-Y Acc (meter/sec**2)"])
ax6b.set_ylabel('B-Y Acc (meter/sec**2)', fontsize=8)
ax6b.set_xlabel('θ12', fontsize=8)
ax6b.grid(True)
ax6b.set_xlim([-3.14, 3.14])

ax7b = f2.add_subplot(3, 3, 7)
ax7b.plot(df["Theta"], df["Magnitude Position B"])
ax7b.set_ylabel('Magnitude Of Position Vector (m)', fontsize=8)
ax7b.set_xlabel('θ12', fontsize=8)
ax7b.grid(True)
ax7b.set_ylim([0.06, 0.13])

ax8b = f2.add_subplot(3, 3, 8)
ax8b.plot(df["Theta"], df["Magnitude Velocity B"])
ax8b.set_ylabel('Magnitude Of Velocity Vector (m/s)', fontsize=8)
ax8b.set_xlabel('θ12', fontsize=8)
ax8b.grid(True)
ax8b.set_ylim([0.0, 0.12])

ax9b = f2.add_subplot(3, 3, 9)
ax9b.plot(df["Theta"], df["Magnitude Acc B"])
ax9b.set_ylabel('Magnitude Of Acceleration Vector (m/s**2)', fontsize=8)
ax9b.set_xlabel('θ12', fontsize=8)
ax9b.grid(True)
ax9b.set_ylim([0.1, 0.6])

plt.suptitle('Kinematic Analysis For Point B', fontsize=20)
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.savefig('outputB.png')

####################################################
f3 = plt.figure(figsize=(15.2, 8.4))

ax1c = f3.add_subplot(3, 3, 1)
ax1c.plot(df["Theta"], df["C-X (meter)"])
ax1c.set_ylabel('C-X (meter)', fontsize=8)
ax1c.set_xlabel('θ12', fontsize=8)
ax1c.grid(True)
ax1c.set_xlim([-3.14, 3.14])

ax2c = f3.add_subplot(3, 3, 4)
ax2c.plot(df["Theta"], df["C-Y (meter)"])
ax2c.set_ylabel('C-Y (meter)', fontsize=8)
ax2c.set_xlabel('θ12', fontsize=8)
ax2c.grid(True)
ax2c.set_xlim([-3.14, 3.14])

ax3c = f3.add_subplot(3, 3, 2)
ax3c.plot(df["Theta"], df["C-X Vel (meter/sec)"])
ax3c.set_ylabel('C-X Vel (meter/sec)', fontsize=8)
ax3c.set_xlabel('θ12', fontsize=8)
ax3c.grid(True)
ax3c.set_xlim([-3.14, 3.14])

ax4c = f3.add_subplot(3, 3, 3)
ax4c.plot(df["Theta"], df["C-X Acc (meter/sec**2)"])
ax4c.set_ylabel('C-X Acc (meter/sec**2)', fontsize=8)
ax4c.set_xlabel('θ12', fontsize=8)
ax4c.grid(True)
ax4c.set_xlim([-3.14, 3.14])

ax5c = f3.add_subplot(3, 3, 5)
ax5c.plot(df["Theta"], df["C-Y Vel (meter/sec)"])
ax5c.set_ylabel('C-Y Vel (meter/sec)', fontsize=8)
ax5c.set_xlabel('θ12', fontsize=8)
ax5c.grid(True)
ax5c.set_xlim([-3.14, 3.14])

ax6c = f3.add_subplot(3, 3, 6)
ax6c.plot(df["Theta"], df["C-Y Acc (meter/sec**2)"])
ax6c.set_ylabel('C-Y Acc (meter/sec**2)', fontsize=8)
ax6c.set_xlabel('θ12', fontsize=8)
ax6c.grid(True)
ax6c.set_xlim([-3.14, 3.14])

ax7c = f3.add_subplot(3, 3, 7)
ax7c.plot(df["Theta"], df["Magnitude Position C"])
ax7c.set_ylabel('Magnitude Of Position Vector (m)', fontsize=8)
ax7c.set_xlabel('θ12', fontsize=8)
ax7c.grid(True)
#ax7c.set_ylim([0.06, 0.13])

ax8c = f3.add_subplot(3, 3, 8)
ax8c.plot(df["Theta"], df["Magnitude Velocity C"])
ax8c.set_ylabel('Magnitude Of Velocity Vector (m/s)', fontsize=8)
ax8c.set_xlabel('θ12', fontsize=8)
ax8c.grid(True)
#ax8c.set_ylim([0.0, 0.12])

ax9c = f3.add_subplot(3, 3, 9)
ax9c.plot(df["Theta"], df["Magnitude Acc C"])
ax9c.set_ylabel('Magnitude Of Acceleration Vector (m/s**2)', fontsize=8)
ax9c.set_xlabel('θ12', fontsize=8)
ax9c.grid(True)
#ax9c.set_ylim([0.0, 0.6])

plt.suptitle('Kinematic Analysis For Point C', fontsize=20)
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.savefig('outputC.png')

####################################################
f4 = plt.figure(figsize=(15.2, 8.4))

ax1d = f4.add_subplot(3, 3, 1)
ax1d.plot(df["Theta"], df["D-X (meter)"])
ax1d.set_ylabel('D-X (meter)', fontsize=8)
ax1d.set_xlabel('θ12', fontsize=8)
ax1d.grid(True)
ax1d.set_xlim([-3.14, 3.14])

ax2d = f4.add_subplot(3, 3, 4)
ax2d.plot(df["Theta"], df["D-Y (meter)"])
ax2d.set_ylabel('D-Y (meter)', fontsize=8)
ax2d.set_xlabel('θ12', fontsize=8)
ax2d.grid(True)
ax2d.set_xlim([-3.14, 3.14])

ax3d = f4.add_subplot(3, 3, 2)
ax3d.plot(df["Theta"], df["D-X Vel (meter/sec)"])
ax3d.set_ylabel('D-X Vel (meter/sec)', fontsize=8)
ax3d.set_xlabel('θ12', fontsize=8)
ax3d.grid(True)
ax3d.set_xlim([-3.14, 3.14])

ax4d = f4.add_subplot(3, 3, 3)
ax4d.plot(df["Theta"], df["D-X Acc (meter/sec**2)"])
ax4d.set_ylabel('D-X Acc (meter/sec**2)', fontsize=8)
ax4d.set_xlabel('θ12', fontsize=8)
ax4d.grid(True)
ax4d.set_xlim([-3.14, 3.14])

ax5d = f4.add_subplot(3, 3, 5)
ax5d.plot(df["Theta"], df["D-Y Vel (meter/sec)"])
ax5d.set_ylabel('D-Y Vel (meter/sec)', fontsize=8)
ax5d.set_xlabel('θ12', fontsize=8)
ax5d.grid(True)
ax5d.set_xlim([-3.14, 3.14])

ax6d = f4.add_subplot(3, 3, 6)
ax6d.plot(df["Theta"], df["D-Y Acc (meter/sec**2)"])
ax6d.set_ylabel('D-Y Acc (meter/sec**2)', fontsize=8)
ax6d.set_xlabel('θ12', fontsize=8)
ax6d.grid(True)
ax6d.set_xlim([-3.14, 3.14])

ax7d = f4.add_subplot(3, 3, 7)
ax7d.plot(df["Theta"], df["Magnitude Position D"])
ax7d.set_ylabel('Magnitude Of Position Vector (m)', fontsize=8)
ax7d.set_xlabel('θ12', fontsize=8)
ax7d.grid(True)
#ax7d.set_ylim([0.06, 0.13])

ax8d = f4.add_subplot(3, 3, 8)
ax8d.plot(df["Theta"], df["Magnitude Velocity D"])
ax8d.set_ylabel('Magnitude Of Velocity Vector (m/s)', fontsize=8)
ax8d.set_xlabel('θ12', fontsize=8)
ax8d.grid(True)
#ax8d.set_ylim([0.0, 0.12])

ax9d = f4.add_subplot(3, 3, 9)
ax9d.plot(df["Theta"], df["Magnitude Acc D"])
ax9d.set_ylabel('Magnitude Of Acceleration Vector (m/s**2)', fontsize=8)
ax9d.set_xlabel('θ12', fontsize=8)
ax9d.grid(True)
#ax9d.set_ylim([0.0, 0.6])

plt.suptitle('Kinematic Analysis For Point D', fontsize=20)
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.savefig('outputD.png')

####################################################
f5 = plt.figure(figsize=(15.2, 8.4))

ax1d0 = f5.add_subplot(3, 3, 1)
ax1d0.plot(df["Theta"], df["D0-X (meter)"])
ax1d0.set_ylabel('D0-X (meter)', fontsize=8)
ax1d0.set_xlabel('θ12', fontsize=8)
ax1d0.grid(True)
ax1d0.set_xlim([-3.14, 3.14])

ax2d0 = f5.add_subplot(3, 3, 4)
ax2d0.plot(df["Theta"], df["D0-Y (meter)"])
ax2d0.set_ylabel('D0-Y (meter)', fontsize=8)
ax2d0.set_xlabel('θ12', fontsize=8)
ax2d0.grid(True)
ax2d0.set_xlim([-3.14, 3.14])

ax3d0 = f5.add_subplot(3, 3, 2)
ax3d0.plot(df["Theta"], df["D0-X Vel (meter/sec)"])
ax3d0.set_ylabel('D0-X Vel (meter/sec)', fontsize=8)
ax3d0.set_xlabel('θ12', fontsize=8)
ax3d0.grid(True)
ax3d0.set_xlim([-3.14, 3.14])

ax4d0 = f5.add_subplot(3, 3, 3)
ax4d0.plot(df["Theta"], df["D0-X Acc (meter/sec**2)"])
ax4d0.set_ylabel('D0-X Acc (meter/sec**2)', fontsize=8)
ax4d0.set_xlabel('θ12', fontsize=8)
ax4d0.grid(True)
ax4d0.set_xlim([-3.14, 3.14])

ax5d0 = f5.add_subplot(3, 3, 5)
ax5d0.plot(df["Theta"], df["D0-Y Vel (meter/sec)"])
ax5d0.set_ylabel('D0-Y Vel (meter/sec)', fontsize=8)
ax5d0.set_xlabel('θ12', fontsize=8)
ax5d0.grid(True)
ax5d0.set_xlim([-3.14, 3.14])

ax6d0 = f5.add_subplot(3, 3, 6)
ax6d0.plot(df["Theta"], df["D0-Y Acc (meter/sec**2)"])
ax6d0.set_ylabel('D0-Y Acc (meter/sec**2)', fontsize=8)
ax6d0.set_xlabel('θ12', fontsize=8)
ax6d0.grid(True)
ax6d0.set_xlim([-3.14, 3.14])

ax7d0 = f5.add_subplot(3, 3, 7)
ax7d0.plot(df["Theta"], df["Magnitude Position D0"])
ax7d0.set_ylabel('Magnitude Of Position Vector (m)', fontsize=8)
ax7d0.set_xlabel('θ12', fontsize=8)
ax7d0.grid(True)
#ax7d0.set_ylim([0.06, 0.13])

ax8d0 = f5.add_subplot(3, 3, 8)
ax8d0.plot(df["Theta"], df["Magnitude Velocity D0"])
ax8d0.set_ylabel('Magnitude Of Velocity Vector (m/s)', fontsize=8)
ax8d0.set_xlabel('θ12', fontsize=8)
ax8d0.grid(True)
#ax8d0.set_ylim([0.0, 0.12])

ax9d0 = f5.add_subplot(3, 3, 9)
ax9d0.plot(df["Theta"], df["Magnitude Acc D0"])
ax9d0.set_ylabel('Magnitude Of Acceleration Vector (m/s**2)', fontsize=8)
ax9d0.set_xlabel('θ12', fontsize=8)
ax9d0.grid(True)
#ax9d0.set_ylim([0.0, 0.6])

plt.suptitle('Kinematic Analysis For Point D0', fontsize=20)
plt.grid(True)
x1, x2, y1, y2 = plt.axis()
plt.axis((-3.14, 3.14, y1, y2))
plt.savefig('outputD0.png')
