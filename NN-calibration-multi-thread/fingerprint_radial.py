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
        self.id = None
        self.style = "radial"  # Default value
        self.atomtypes = None
        self.empty = True      # Default value
        self.fullydefined = False  # Default value
        self.inputpath = inputpath

    def parser(self):
        with open(self.inputpath, 'r') as file:
            metaparams = {}  # Dictionary to store key-value pairs
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

Zinc = Fingerprint_radial("/Users/andrewtrepagnier/.cursor-tutor/research/Py-Calibration/Zn.input")
Zinc.parser()



        # self.full_filename_path = 
        # input_lines = 
        # for line in input_lines:






