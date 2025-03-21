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

    def __init__(self, n_body_type, dr, re, rc, alpha, n, o, id, style=None, atomtypes=None, empty=True, fullydefined = False, inputpath=None):

        self.n_body_type = n_body_type
        self.dr = dr
        self.re = re
        self.rc = rc
        self.alpha = np.array([alpha])
        self.n = n
        self.o = o
        self.id = id
        self.style = "radial" #Default to "radial"
        self.atomtypes = np.array([n_body_type])
        self.empty = empty
        self.fullydefined = fullydefined
        self.inputpath = inputpath

    def parser(self):
        with open(self.inputpath, 'r') as file:
            metaparams = {}  # python dictionary
            current_label = None
            
            for line_num, line in enumerate(file, 1): # gets rid of the white spaces, saves line numbers for splicing, and splits key and vals
                line = line.strip() 
                if not line:
                    continue
                # Split the line by ':' 
                parts = line.split(':')
                
                if len(parts) > 1:  # This is a label line
                    # Use the last non-empty part as the key
                    key = parts[-2].strip() if len(parts) > 2 else parts[-1].strip()
                    value_str = parts[-1].strip()

                #   Counts number of lines and gets rid of white spaces for better readability in the next for loop
                # elif type(line) == float:
                #     line = metaparams[line_num] # save that metaparameter in an array
                # elif type(line) == str:
                #     line = metaparam_labels[line_num]

                # line_num = 0
                # for line in file:
                #     line = line.strip()
                #     if not line:
                #         continue
                #     line_num += 1
                # # At this point, input file has no white space and line_num(the line number) is accurate
                #     if line_num >= 11 && line_num <=24:
                #         key = line
                
                #     line = line.strip()
                #     if not line:
                #         continue
                #     terms = line.split(':')
                #     # Some lines of the input file have multiple terms split by a colon. the if statement
                #     #handles cases where there are multiple splits for so there is always a key and value.
                #     if len(terms) > 1:

                    # Try to convert to float(s) if possible
                    try:
                        # Check if multiple values on the line
                        values = [float(v) for v in value_str.split()]
                        # If only one value, don't keep it as a list
                        metaparams[key] = values[0] if len(values) == 1 else values
                    except ValueError:
                        # If conversion fails, store as string
                        metaparams[key] = value_str
                else:
                    # This is a continuation line or value without label
                    if current_label and line:
                        try:
                            # Try to convert to float(s)
                            values = [float(v) for v in line.split()]
                            metaparams[current_label] = values[0] if len(values) == 1 else values
                        except ValueError:
                            metaparams[current_label] = line

            # Now you can access values like:
            print(metaparams['re'])  # instead of metaparams['fingerprintconstants:Zn_Zn:radialscreened_0:re']

        self.full_filename_path = 
        input_lines = 
        for line in input_lines:





