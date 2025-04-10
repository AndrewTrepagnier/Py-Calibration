from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Bondparameters:

    re : float
    rc : float
    m : int
    k : int
    alphaks : List[float]

@dataclass
class AtomicSystem:

    """Storing our atomic configuration parameters"""

    num_atoms : int
    atom_positions : List[float]
    energy : List[float]
    distance_matrix : List[float]
    fingerprints : List[float]


class Fingerprint_bond:

    def __init__(self, inputpath: Optional[str] = None, dumppath: Optional[str] = None):
        self.params : Optional[Bondparameters] # instance variable self.params declared, the next part is a type of hint telling python that this variable can either be Bondparamters object or None
        self.systems : List[AtomicSystem] #List in which each element is an AtomicSystem object. Each atomic system object will have thiose attributes that is defined in the dataclass.
        """
        Self.system = [
        AtomicSystem(
        num_atoms = 52,
        atom_positions = [....],
        energy = [...],
        distance_matrix = [....],
        fingerprints=[...]),

        AtomicSystem(
        .
        .
        .)
        
        ]
        Then, if you wanted to call on the first system, you could do so like this:
        first_system = self.system[0]

        """
        self.style = "bond"
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
        

        # Initialize parameter values, open the input file, read the content as lines
        params_dict = {}                            #empty dictionary to store parameters being extracted from the input file
        with open(self.inputpath, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):             #loops through lines by index
                line = lines[i].strip()             #removes white spaces
                if not line:                        #skips empty lines
                    continue

        
                terms = line.split(':')             #split lines by colon
                if len(terms) > 2:                  # loop that only takes the second to last term in the line, since that is what our parameter key is
                    param_name = terms[-2].strip()  
                    value = lines[i+1].strip()      #The value is on the line after the key



                    if param_name in ['re', 'rc', 'dr', 'n', 'o', 'm', 'k']:    #turn the strings to floats
                        params_dict[param_name] = float(value)


                    elif param_name in ['alphaks']:                      
                        string_list = value.split()             #value.split() takes a sting like 1.0 2.0 3.0 and splits it into a list of strings like - ["1.0", "2.0", "3.0"]
                        values = []                             #then float(x) for x in value.split() converts each sting, x, from the split list to a float.                     
                        for x in string_list:
                            float_number = float(x)
                            values.append(float_number)
                        params_dict[param_name] = values



        # Create RadialParameters instance
        self.params = Bondparameters(
            re=params_dict.get('re'),
            rc=params_dict.get('rc'),
            dr=params_dict.get('dr'),
            # alpha=params_dict.get('alpha', []),
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

        print(f"Found dump files: {dump_files}")  # Debug print
        
        for filename in dump_files:
            with open(os.path.join(self.dumppath, filename), "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            print(f"First few lines of {filename}:")  # Debug print
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {line.strip()}")  # Debug print

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
            distance_matrix = np.zeros((num_atoms, num_atoms))
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        distance_matrix[i,j] = np.linalg.norm(positions[i] - positions[j])

            # Create and store new AtomicSystem
            system = AtomicSystem(
                num_atoms=num_atoms,
                atom_positions=positions,
                box_bounds=box_bounds,
                energy=energy,
                distance_matrix=distance_matrix
            )
            self.systems.append(system)
        print(f"The energy of the system is {energy}")

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

    def radii_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        r1 = np.zeros(res + buffer) #Vector that stores the each radius from 0 to rc with res points inbetween.
        
        for k in range(res + buffer):
            r1[k] = self.params.rc**2 * k/res 
            r_sqrt = np.sqrt(r1[k])
            
            # Once we have our vector of r1 values between 0 and rc, compute radial functions for each m value using the sqrt(r1[k]) value of this each iteration
            # This function will give an exponential decay component to the radii.
            for m in range(int(self.params.n - self.params.o)):
                radial_function = (
                    (r_sqrt/self.params.re)**(m + self.params.o) * 
                    np.exp(-self.params.alphak[m] * (r_sqrt/self.params.re)) * 
                    self.cutoff_function(r_sqrt)
                )
                radial_table[k] = radial_function #Think of this as the table of dependent variables to the table of independent variables, r1

            # Compute cutoff function derivatives
            if r_sqrt >= self.params.rc or r_sqrt <= self.params.rc - self.params.dr:
                dfctable[k] = 0
            else:
                term = (self.params.rc - r_sqrt) / self.params.dr
                dfctable[k] = (-8 * (1 - term)**3) / (self.params.dr * (1 - term)**4)

        return r1, radial_table, dfctable # Added r1 to the table since this is what each element of the distance matrix will look at first and see where it lies.






