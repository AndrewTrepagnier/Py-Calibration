import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from fingerprint_radial import Fingerprint_radial 

"""
Edit the json file to test different filepaths or parameters. We can manually input the parameters we know our fingerprint should be producing 
and run it through this testing script to see if the fingerprint is computing and parsing the way we desire.
"""

def load_test_config(config_files = "test_config1.json"):
    """Loads the test configuration from the json file"""
    with open(config_files, 'r') as f:
        return json.load(f)


def make_fingerprint_instance():
    # load test configuration
    config = load_test_config()
    #initialize fingerprint calculator
    Zn_fingerprint = Fingerprint_radial(inputpath=config['input_file'],
                                      dumppath=config['dump_file'])
    return Zn_fingerprint, config



def test_input_parser():
    Zn_fingerprint, config = make_fingerprint_instance()

    Zn_fingerprint.input_parser()
    assert abs(Zn_fingerprint.params.rc - config['rc'])<1e-4, \
    f"cutoff radius {Zn_fingerprint.params.rc} doesn't match expected {config['rc']}"



def test_output_parser():
    Zn_fingerprint, config = make_fingerprint_instance()

    Zn_fingerprint.dump_parser()
    # Can we save our data in system
    assert len(Zn_fingerprint.systems) > 0, "No systems loaded"
    # What is the number of atoms that was parsed
    assert Zn_fingerprint.systems[0].num_atoms == config['num_atoms'], \
    f"Number of atoms {Zn_fingerprint.systems[0].num_atoms} doesn't match the expected to be {config['num_atoms']}"
    # What is the energy
    assert abs(Zn_fingerprint.systems[0].energy - config['energy']) < 1e-4, \
    f"The energy parsed in the dump file, {Zn_fingerprint.systems[0].energy} doesn't match the {config['energy']}"



def test_distance():
    Zn_fingerprint, config = make_fingerprint_instance()
    
    Zn_fingerprint.dump_parser()
    atom1 = config['test_distance']['atom1']
    atom2 = config['test_distance']['atom2']
    computed_distance = Zn_fingerprint.systems[0].distance_matrix[atom1, atom2] # computes the euclidian distance between atom 0 and atom 1 in the matrix we store their coordinate positions in
    expected_distance = config['test_distance']['expected_distance'] # calculated the desired by hand, but an equation could be inserted here if need be
    
    assert abs(computed_distance - expected_distance) < 1e-4, \
        f"Distance between atoms {atom1} and {atom2} is {computed_distance}, " \
        f"but expected {expected_distance}"
    
def call_fingerprint_computation():
    Zn_fingerprint, config = make_fingerprint_instance()
    Zn_fingerprint.input_parser()
    Zn_fingerprint.dump_parser()
    Zn_fingerprint.radii_table()
    Zn_fingerprint.compute_fingerprint()


if __name__ == "__main__":
    print("Running parameter test...")
    test_input_parser()
    print("Test passed! ✅")
    
    print("\nRunning output parser test...")
    test_output_parser()
    print("Test passed! ✅")

    print("Running ij pair testing...")
    test_distance()
    print("Test passed! ✅")

    print("Computing fingerprints...")

    call_fingerprint_computation()
    print("Test passed! ✅")