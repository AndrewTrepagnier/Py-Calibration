"""
    Transcribed python version of the bond structural fingerprint for a pytorch/tensorflow network implementation.
    This file reads the DFT dump file data in a dictionary format and computes fingerprints with a given set of metaparameters

    ------------------------------------------------------------------------------------------
    Contributing Author:  Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com | 4/7/2025
    ------------------------------------------------------------------------------------------
"""


from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import os

@dataclass
class BondParameters:
    