# 570-Final-Project

System Requirements
Linux (Tested on Ubuntu 18.04.05)/ Windows

Python (Tested with v3.7)
Python Requirements
SciPy 1.1.0
Matplotlib 3.0.0
NumPy 1.15.2
imageio
math





There are different sphereworld that are in the root folder, with different excersises the patient has to perform.

DATA_ROOT_DIR/
└──── sphereworld1.mat
└──── sphereworld2.mat
└──── sphereworld3.mat
└──── sphereworld4.mat


*** change the sphereworld directory in main.py and potential_final.py before you run the scripts***

To generate the results use 2 scripts****************************************************************

Run main.py for generation a simulated safe path, and a gif of the parallel manipulator moving is generated. ( change the sphereworld directory before you run the scripts)

Output is a GIF file that is generated in the root folder.
******************************************************************************************************
Run potential_final.py is used to calculate the path, and it has three functions attractive, repulsive potentials and navigation potential.
And it also generates a path.

******************************************************************************************************


OTHER SCRIPTS


geometry_final.py has a support function for generating different shapes for the simulation.


Linkage.py,Neckbrace.py is used to calculate the inverse kinematics.
pressure.py gets the pressure sensor readings from Arduino.
