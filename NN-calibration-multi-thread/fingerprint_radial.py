
"""
    Transcribed python version of the radial bond structural fingerprint for a pytorch/tensorflow network implementation.
    This file reads the DFT dump file data in a dictionary format and computes fingerprints with a given set of metaparameters

    ------------------------------------------------------------------------------
    Contributing Author:  Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com 
    ------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import os

@dataclass
class RadialParameters:
    """
    Stores and validates all parameters for radial fingerprint computation.
    """
    re: float              # Equilibrium distance
    rc: float             # Cutoff radius
    dr: float            # Radial step size
    alpha: List[float]   # Alpha parameters
    alphak: List[float]  # Alphak parameters
    n: float            # Power series upper bound
    o: float           # Power series lower bound
    m: Optional[float] = None
    k: Optional[float] = None

    def __post_init__(self):
        """Validates parameters after initialization"""
        assert self.rc > 0, "Cutoff radius must be positive"
        assert self.dr > 0, "Radial step must be positive"
        assert len(self.alpha) > 0, "Must provide at least one alpha value"

@dataclass
class AtomicSystem:
    """
    Represents a complete atomic system configuration.
    Stores both structural data and computed properties.
    """
    num_atoms: int
    atom_positions: np.ndarray    # Shape: (n_atoms, 3) for x,y,z coordinates
    box_bounds: np.ndarray       # Shape: (3, 2) for min/max in x,y,z
    energy: float
    distances: Optional[np.ndarray] = None  # Pairwise distances matrix
    fingerprints: Optional[np.ndarray] = None

class Fingerprint_radial:
    def __init__(self, inputpath: Optional[str] = None, dumppath: Optional[str] = None):
        self.params: Optional[RadialParameters] = None
        self.systems: List[AtomicSystem] = []
        self.style = "radial"
        self.inputpath = inputpath
        self.dumppath = dumppath

    def input_parser(self):
        """
        Parse through looking for metaparameters and assign their corresponding values.
        Note: this parsing function will NOT work properly if input scripts do not adhere 
        to the same structure as Zn.input here: https://github.com/ranndip/Calibration.git
        """
        if not self.inputpath:
            raise ValueError("Input path not specified")

        # Initialize parameter values
        params_dict = {}
        
        with open(self.inputpath, 'r') as file:
            lines = file.readlines()
            
            for i in range(len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                terms = line.split(':')
                if len(terms) > 2:
                    param_name = terms[-2].strip()
                    value = lines[i+1].strip()

                    if param_name in ['re', 'rc', 'dr', 'n', 'o', 'm', 'k']:
                        params_dict[param_name] = float(value)
                    elif param_name in ['alpha', 'alphak']:
                        values = [float(x) for x in value.split()]
                        params_dict[param_name] = values

        # Create RadialParameters instance
        self.params = RadialParameters(
            re=params_dict.get('re'),
            rc=params_dict.get('rc'),
            dr=params_dict.get('dr'),
            alpha=params_dict.get('alpha', []),
            alphak=params_dict.get('alphak', []),
            n=params_dict.get('n'),
            o=params_dict.get('o'),
            m=params_dict.get('m'),
            k=params_dict.get('k')
        )

        # Print parameters for verification
        print(f"Loaded parameters:")
        for key, value in params_dict.items():
            print(f"{key} is {value}")

    def dump_parser(self):
        """
        Parse LAMMPS dump files and create AtomicSystem instances.

       
        Typical LAMMPS dump item should look like this, if not the parsing algorithms will be incorrect:

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
        if not self.dumppath:
            raise ValueError("Dump path not specified")

        dump_files = [f for f in os.listdir(self.dumppath) if os.path.isfile(os.path.join(self.dumppath, f))]
        
        if not dump_files:
            print("No dump files in folder location specified")
            return

        for filename in dump_files:
            with open(os.path.join(self.dumppath, filename), "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse number of atoms
            num_atoms = int(lines[3].strip())

            # Parse energy
            energy = float(lines[1].split()[1])

            # Parse atomic positions
            positions = np.zeros((num_atoms, 3))
            for i in range(num_atoms):
                line_data = lines[9+i].split()
                positions[i] = [float(x) for x in line_data[2:5]]

            # Parse box bounds (simplified - you might need to adjust this)
            box_bounds = np.zeros((3, 2))
            for i in range(3):
                bounds = lines[5+i].split()
                box_bounds[i] = [float(bounds[0]), float(bounds[1])]

            # Create distance matrix
            """
            This will look like this:

                    atom0   atom1   atom2   ....

            atom0   i=j=None  [0][1]  [0][2] 

            atom1   [1][0]   i=j=None   [1][2]

            atom2   .           .           .
            
            """
            distances = np.zeros((num_atoms, num_atoms))
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        distances[i,j] = np.linalg.norm(positions[i] - positions[j])

            # Create and store new AtomicSystem
            system = AtomicSystem(
                num_atoms=num_atoms,
                atom_positions=positions,
                box_bounds=box_bounds,
                energy=energy,
                distances=distances
            )
            self.systems.append(system)

    def cutoff_function(self, r: float) -> float:
        """
        Compute the cutoff function for a given radius.
        
        Args:
            r (float): Distance for which to compute cutoff function
            
        Returns:
            float: Cutoff function value between 0 and 1
        """
        if not self.params:
            raise ValueError("Parameters not initialized. Run input_parser first.")
            
        x = (self.params.rc - r) / self.params.dr
        if x > 1:
            return 1
        elif 0 <= x <= 1:
            return (1 - (1 - x)**4)**2  # continuous cutoff range for smoother potentials
        else:
            return 0

    def radii_table(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate interpolation tables of radii and derivatives for computational efficiency.
        
        Returns:
            tuple: (radial_table, dfctable) containing precomputed values
        """
        if not self.params:
            raise ValueError("Parameters not initialized. Run input_parser first.")

        buffer = 5
        res = 1000

        radial_table = np.zeros(res + buffer)
        dfctable = np.zeros(res + buffer)
        r1 = np.zeros(res + buffer)

        for k in range(res + buffer):
            r1[k] = self.params.rc**2 * k/res
            r_sqrt = np.sqrt(r1[k])
            
            # Compute radial functions for each m value
            for m in range(int(self.params.n - self.params.o)):
                radial_function = (
                    (r_sqrt/self.params.re)**(m + self.params.o) * 
                    np.exp(-self.params.alphak[m] * (r_sqrt/self.params.re)) * 
                    self.cutoff_function(r_sqrt)
                )
                radial_table[k] = radial_function

            # Compute cutoff function derivatives
            if r_sqrt >= self.params.rc or r_sqrt <= self.params.rc - self.params.dr:
                dfctable[k] = 0
            else:
                term = (self.params.rc - r_sqrt) / self.params.dr
                dfctable[k] = (-8 * (1 - term)**3) / (self.params.dr * (1 - term)**4)

        return radial_table, dfctable

    def compute_fingerprint(self, system_index: int = 0) -> np.ndarray:
        """
        Compute fingerprints for a given atomic system.
        
        Args:
            system_index (int): Index of the system to compute fingerprints for
            
        Returns:
            np.ndarray: Computed fingerprints for each atom
        """
        if not self.systems:
            raise ValueError("No atomic systems loaded. Run dump_parser first.")
        
        system = self.systems[system_index]
        
        # If fingerprints already computed, return cached version
        if system.fingerprints is not None:
            return system.fingerprints
            
        # Initialize fingerprint array
        n_fingerprints = len(self.params.alpha)  # or however you determine this
        fingerprints = np.zeros((system.num_atoms, n_fingerprints))
        
        # Pre-compute radii table for efficiency
        radial_table, dfctable = self.radii_table()
        
        # Compute fingerprints using vectorized operations where possible
        for i in range(system.num_atoms):
            neighbor_distances = system.distances[i]
            valid_neighbors = neighbor_distances < self.params.rc
            
            # Your fingerprint computation logic here
            # Use vectorized operations with numpy where possible
            # Example:
            # fingerprints[i] = some_vectorized_operation(neighbor_distances[valid_neighbors])
            
        system.fingerprints = fingerprints
        return fingerprints