"""
    Transcribed python version of the radial bond structural fingerprint for a pytorch/tensorflow network implementation.
    This file reads the DFT dump file data in a dictionary format and computes fingerprints with a given set of metaparameters

    ------------------------------------------------------------------------------
    Contributing Author:  Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com 
    ------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import os

class Fingerprint_radial:

    #def __init__(self, n_body_type, dr, re, rc, alpha, alphak, n, o, id, style=None, atomtypes=None, empty=True, fullydefined = False, inputpath=None):
    def __init__(self, inputpath=None, dumppath=None):
        
        self.n_body_type = None
        self.dr = None
        self.re = None
        self.rc = None
        self.alpha = []
        self.alphak = None
        self.n = None
        self.o = None
        self.m = None
        self.k = None
        self.id = None
        self.style = "radial"  # Default value
        self.atomtypes = None
        self.empty = True      # Default value
        self.fullydefined = False  # Default value
        self.inputpath = inputpath
        self.dumppath = dumppath

    def input_parser(self):

        """
        Parse through looking for metaparameters and assign their corresponding values as objects of self. (ex: for Zinc, it may assignn self.re = 2.6 based off a tranditional input file(must be txt file)
        Note: this parsing function will NOT work properly if input scripts do not adhere to the same structure as Zn.input here: https://github.com/ranndip/Calibration.git
        """

        with open(self.inputpath, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
        
            for i in range(len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                terms = line.split(':')
                # creates an array of numbers counting up to the number of lines and a list of strings
                if len(terms) > 2:  # Add this line to check length
                    str = terms[-2].strip() # Saves the last term as str, this will look like re, rc, dr, ect.


                    if str == "re":
                    # metaparams['re'] = float(lines[i+1].strip()) # save the next line item(string) as a float and the value of the key in the metaparams dictionary
                        self.re = float(lines[i+1].strip())

                    elif str == "rc":
                    #  metaparams['rc'] = float(lines[i+1].strip())
                        self.rc = float(lines[i+1].strip())

                    elif str == "alpha":
                        # n_alphas is initially a string containing all numbers
                        n_alphas = lines[i+1].strip()  # Ex. "6.950000 6.950000 6.950000 6.950000 6.950000"
                        # Split the string into a list of strings
                        alpha_values = n_alphas.split()  # Ex. ['6.950000', '6.950000', '6.950000', '6.950000', '6.950000']
                        alpha_floats = [float(alpha) for alpha in alpha_values] # Convert each string to float and store in list
                        self.alpha = alpha_floats  

                    elif str == "dr":
                        self.dr = float(lines[i+1].strip())

                    elif str == "n":
                        self.n = float(lines[i+1].strip())

                    elif str == "o":
                        self.o = float(lines[i+1].strip())
                    elif str == "m":
                        self.m = float(lines[i+1].strip())
                    elif str == "k":
                        self.k = float(lines[i+1].strip())
                    elif str == "alphak":
                        # n_alphas is initially a string containing all numbers
                        n_alphaks = lines[i+1].strip()  # Ex. "6.950000 6.950000 6.950000 6.950000 6.950000"
                        # Split the string into a list of strings
                        alphak_values = n_alphaks.split()  # Ex. ['6.950000', '6.950000', '6.950000', '6.950000', '6.950000']
                        alphak_floats = [float(k) for k in alphak_values] # Convert each string to float and store in list
                        self.alphak = alphak_floats  

        print(f"re is {self.re}")
        print(f"rc is {self.rc}")
        print(f"alpha is {self.alpha}")
        print(f"dr is {self.dr}")
        print(f"n is {self.n}")
        print(f"o is {self.o}")
        print(f"alphak is {self.alphak}")

    def dump_parser(self):

        """
        Typical LAMMPS dump item:

        ITEM: TIMESTEP energy, energy_weight, force_weight, nsims
        1        -199944.4130284513230436    1    1   88
        ITEM: NUMBER OF ATOMS
        52        
        ITEM: BOX BOUNDS xy xz yz pp pp pp
        -6.6599068561068142     10.6426540787132993     -5.7964044747763319
        -1.5903743695984427      8.9256853388485613     -0.8635023813304824
        0.0000000000000000     16.3996700907443120     -1.5903743695984427
        ITEM: ATOMS id type x y z
        1      1          1.5004620351452496     0.5243795304393224     3.8834927664029966
        2      1         -0.4657385064656120     1.2308782860044023     7.0568980742855700
        3      1          1.2745152962579427     0.0833213387822464     9.3340024694978840
        .       .           .
        .       .           .
        .       .           .
        52     1          4.0887810494581123     5.4317594751390095    15.0272327985735377

    
        """

        list_of_dump_files = [f for f in os.listdir(self.dumppath) if os.path.isfile(os.path.join(self.dumppath, f))]
        num_files = len(list_of_dump_files)
        dump_contents = {} #this is what's in each dump file, which should be like the commented out section above

        # open and save the content in each file to dump_contents
        if num_files == 0:
            print("No dump files in folder location specified")
        else:   
            for file in list_of_dump_files:
                file_path = os.path.join(self.dumppath, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    dump_contents[file] = f.readlines()  # Changed to readlines() instead of read()

        for dump_data in dump_contents.items():
            filename, lines = dump_data  # Unpack the filename and content

            # Get number of atoms from the file
            num_atoms = int(lines[3].strip())  # Assuming NUMBER OF ATOMS is always line 4
            num_lines_in_item = num_atoms + 9  # Total number of lines of data recorded for each timestep
            
            # Initialize arrays correctly
            atom_line = np.zeros((num_atoms, 4))  # Fixed array creation
            atom_position = np.zeros((num_atoms, 3))  # Will store just the x,y,z coordinates
            atom_matrix = np.empty((num_atoms, num_atoms))  # Matrix for pairwise distances
            energy = []

            # Get energy from first line
            energy.append(float(lines[1].split()[1]))

            # Read atomic positions
            for i in range(num_atoms):
                line_data = lines[9+i].split()  # ITEM: ATOMS starts at line 9
                id = int(line_data[0])
                xal, yal, zal = map(float, line_data[2:5])  # Get x,y,z coordinates
                atom_line[i] = [id, xal, yal, zal]
                atom_position[i] = [xal, yal, zal]
            
            # Calculate pairwise distances
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        Euclid_displacement = np.linalg.norm(atom_position[i] - atom_position[j])
                        atom_matrix[i,j] = Euclid_displacement
                    else:
                        atom_matrix[i,j] = None  # Same atom case
            
            # Store results as class attributes
            self.atom_positions = atom_position
            self.distances = atom_matrix
            self.energies = energy

    def cutoff_function(self, r): #Beauty of having a class-based function in python is that rc, dr are saved in self at all times
            x = (self.rc - r)/self.dr
            if x > 1:
                return 1
            elif 0 <= x <= 1:
                return (1 - (1 - x)**4)**2 #continuous cutoff range for smoother potentials
            else:
                return 0
    
    def radii_table(self):
        """
        Interpolation tables of radii and derivatives for computational efficiency

        """
        buffer = 5
        res = 1000

        radial_table = np.array[res+buffer]
        dfctable = np.array[res+buffer] 
        r1 = []
        # m and k give control on the number of fignerprints you want in the input layer

        for k in range(res+buffer): #k is a distance point
            r1.append(self.rc**2 * k/res)
            """ For m starting at 0, if m is less than or equal to n-o+1 then excecute the line and evaluate m = 1 ...."""
            for m in range(self.n-self.o): #m is a power from 0 to n-o
                #The radial function is the fingerprint calulation. When we are within the cutoff radius, this will be a non zero number that is tabluated in radial_table. When it is outside the cutoff radius, it will be a zero in the table
                radial_function = (np.sqrt(r1[k])/self.re)**(m+self.o) * np.exp(-self.alphak[m]*(np.sqrt(r1[k])/self.re)) * Fingerprint_radial.cutoff_function(np.sqrt(r1[k])) 

            radial_table.append(radial_function)
            if np.sqrt(r1[k]) >= self.rc or np.sqrt(r1[k]) <= self.rc-self.dr: # Cutoff function derivatives
                dfctable.append(0)
            else:
                dfctable.append( (-8* ( 1 - (self.rc - np.sqrt[k])/self.dr)**3) / self.dr / (1 - (self.rc - np.sqrt[k])/self.dr)**4 )

        
    def compute_fingerprint(self):
        """
        LAMMPS can read dump files and build the neighbor lists for each atom. 
        This function recieves the pre-calculated neighbor lists and each atoms relative positions(xn, yn, zn) as well as the number of neighbors, jnum
        In other words, this function processes data in this sequence: 
        For each atom i:
            → Gets its neighbors (jnum)
            → Calculates fingerprints using neighbor distances
            → Updates features and their derivatives
        
        """
        features = []
        dfeaturesx = []
        dfeaturesy = []
        dfeaturesz = []







        







              
            



    

# Zinc = Fingerprint_radial("/Users/andrewtrepagnier/.cursor-tutor/research/Py-Calibration/Zn.input")
# Zinc.parser()



    
