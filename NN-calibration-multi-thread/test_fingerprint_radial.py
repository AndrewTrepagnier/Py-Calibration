import json
import numpy as np
from fingerprint_radial import Fingerprint_radial 

"""
Edit the json file to test different filepaths or parameters. We can manually input the parameters we know our fingerprint should be producing 
and run it through this testing script to see if the fingerprint is computing and parsing the way we desire.
"""

def load_test_config(config_files = "test_config.json"):
    """Loads the test configuration from the json file"""
    with open(config_files, 'r') as f:
        return json.load(f)
    
def test_fingerprint():
    # load test configuration
    config = load_test_config()

    #initialize fingerprint calculator
    Zn_fingerprint = Fingerprint_radial(inputpath=config['input_file'],
                            dumppath=['dump_file'])
    
    # I called this class instance Zn_fingerprint because I first tested Zinc on this
    
    Zn_fingerprint.input_parser()
    assert abs(Zn_fingerprint.params.rc - config['rc'])<1e-4, \
    f"cutoff radius {Zn_fingerprint.params.rc} doesn't match expected {config['re']}"

    Zn_fingerprint.dump_parser()
    assert len(Zn_fingerprint.systems) > 0, "No systems loaded"
    assert Zn_fingerprint.systems[0].num_atoms == config['num_atoms'], \
    f"Number of atoms {Zn_fingerprint.systems[0].num_atoms} doesn't match the expected to be {config['num_atoms']}"

    assert abs(Zn_fingerprint.systems[0].energy) - config['energy'] > 1e-4, \
    f"The energy parsed in the dump file, {Zn_fingerprint.systems[0].energy} doesn't match the {config['energy']}"


    Zn_fingerprint.radii_table()
    Zn_fingerprint.compute_fingerprint()
    
    # Get the distance between two specific atoms
    atom1 = config['test_distance']['atom1']
    atom2 = config['test_distance']['atom2']
    computed_distance = Zn_fingerprint.systems[0].distances[atom1, atom2]
    expected_distance = config['test_distance']['expected_distance']
    
    assert abs(computed_distance - expected_distance) < 1e-4, \
        f"Distance between atoms {atom1} and {atom2} is {computed_distance}, " \
        f"but expected {expected_distance}"
    
    assert Zn_fingerprint.systems[0]

