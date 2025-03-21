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
            metaparams = {}  # Using a dictionary instead of separate arrays
            current_label = None
            
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                # Split the line by ':' to separate labels and values
                parts = line.split(':')
                
                if len(parts) > 1:  # This is a label line
                    current_label = ':'.join(parts[:-1])  # Handle cases with multiple ':'
                    value_str = parts[-1].strip()
                    
                    # Try to convert to float(s) if possible
                    try:
                        # Check if multiple values on the line
                        values = [float(v) for v in value_str.split()]
                        # If only one value, don't keep it as a list
                        metaparams[current_label] = values[0] if len(values) == 1 else values
                    except ValueError:
                        # If conversion to float fails, store as string
                        metaparams[current_label] = value_str
                else:
                    # This is a continuation line or value without label
                    if current_label and line:
                        try:
                            # Try to convert to float(s)
                            values = [float(v) for v in line.split()]
                            metaparams[current_label] = values[0] if len(values) == 1 else values
                        except ValueError:
                            # If conversion fails, store as string
                            metaparams[current_label] = line

        self.full_filename_path = 
        input_lines = 
        for line in input_lines:





