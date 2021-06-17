/*
 * main.cpp
 *
 *  Created on: Jul 22, 2020
 *      Author: kip
 */

/* TO DO list of functions:
 *
 * 1) main
 * 2) read_calibration_parameters
 * 3) read_potential_file
 * 4) generate_random_parameters
 * 5) read_dump_file
 * 6) build_neighbor_list
 * 7) compute
 * 8) create_jacobian
 * 9) levenburg_marquet
 * 10) fast_matrix_division
 *
 */

/* TO DO other
 *
 * 0) doforce flags in fingerprints?
 * 1) Add screening options to potential file
 * 2) Make screening type dependent
 * 3) Optimize screening
 * 4) Migrate all identical functions out of pair_nn_lammps
 * 5) self-contain everything in pair_nn_lammps needed by activations and fingerprints
 * 6) Magnetic radial fingerprint
 * 7) Torsion fingerprint
 * 8) Screened magnetic and torsion
 *
 */

//read arguments

//read potential file


#include "pair_rann.h"

using namespace LAMMPS_NS;

//read command line input.
int main(int argc, char **argv)
{
	char str[MAXLINE];
	if (argc!=3 || strcmp(argv[1],"-in")!=0){
		sprintf(str,"syntax: nn_calibration -in \"input_file.nn\"\n");
		std::cout<<str;
	}
	else{
		PairRANN *cal = new PairRANN(argv[2]);
		cal->setup();
		cal->run();
		cal->finish();
		delete cal;
	}
}

