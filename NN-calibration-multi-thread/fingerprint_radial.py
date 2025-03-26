"""
    Transcribed python version of the radial bond structural fingerprint for a pytorch/tensorflow network implementation.
    This file reads the DFT dump file data in a dictionary format and computes fingerprints with a given set of metaparameters

    ------------------------------------------------------------------------------
    Contributing Author:  Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com 
    ------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

class Fingerprint_radial:

    #def __init__(self, n_body_type, dr, re, rc, alpha, alphak, n, o, id, style=None, atomtypes=None, empty=True, fullydefined = False, inputpath=None):
    def __init__(self, inputpath=None):
        
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

    def parser(self):

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

        radial_table = np.array(res+buffer)
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
        
        








    
            
    
    

              
            



    

Zinc = Fingerprint_radial("/Users/andrewtrepagnier/.cursor-tutor/research/Py-Calibration/Zn.input")
Zinc.parser()



        # self.full_filename_path = 
        # input_lines = 
        # for line in input_lines:






