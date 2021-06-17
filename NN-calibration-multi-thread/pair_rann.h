/*
 * calibration.h
 *
 *  Created on: Jul 27, 2020
 *      Author: kip
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <map>
#include <dirent.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <sys/resource.h>
#define MAXLINE 4096
#define SHORTLINE 128
#define NEIGHMASK 0x3FFFFFFF

#ifndef CALIBRATION_H_
#define CALIBRATION_H_

namespace LAMMPS_NS{

class PairRANN{
public:
	PairRANN(char *);
	~PairRANN();
	void setup();
	void run();
	void finish();
	void read_potential_file();
	void read_atom_types(char **,char *);
	void read_mass(char **,char *);
	void read_fpe(char**,char *);//fingerprints per element. Count total fingerprints defined for each 1st element in element combinations
	void read_fingerprints(char **,int,char *);
	void read_fingerprint_constants(char **,int,char *);
	void read_network_layers(char**,char*);//include input and output layer (hidden layers + 2)
	void read_layer_size(char**,char*);
	void read_weight(char**,char*,FILE*);//weights should be formatted as properly shaped matrices
	void read_bias(char**,char*,FILE*);//biases should be formatted as properly shaped vectors
	void read_activation_functions(char**,char*);
	void read_screening(char**,int, char*);
	void read_parameters(char**,char*);
	void allocate(char **);//called after reading element list, but before reading the rest of the potential
	bool check_parameters();
	bool check_potential();
	void read_dump_files();
	void create_neighbor_lists();
//	void screen(double*,double*,double*,double*,double*,double*,double*,bool*,int,int);
	void screen(double*,double*,double*,double*,double*,double*,double*,bool*,int,int,double*,double*,double*,int *,int);
	void create_random_weights(int,int,int,int);
	void create_random_biases(int,int,int);
	void compute_fingerprints();
	void separate_validation();
	void update_stack_size();
	void cull_neighbor_list(double *,double *,double *,int *,int *,int *,int,int);
	void screen_neighbor_list(double *,double *,double *,int *,int *,int *,int,int,bool*,double*,double*,double*,double*,double*,double*,double*);



	void levenburg_marquardt_qr();
	void levenburg_marquardt_ch();
	void levenburg_marquardt_ch_g();
	void conjugate_gradient();
	void write_potential_file(bool,char*);
	void errorf(const char *);
	int count_words(char *);


	//parameters
	char *algorithm;
	char *potential_input_file;
	char *dump_directory;
	bool doforces;
	double tolerance;
	double regularizer;
	bool doregularizer;
	char *log_file;
	char *potential_output_file;
	int potential_output_freq;
	int max_epochs;
	//char* dims_reserved_temp1;
	//char* dims_reserved_temp2;
	int nsims;
	int nsets;
	int betalen;
	int jlen1;
	int *betalen_v;
	int natoms;
	int natomsr;
	int natomsv;
	double validation;
	int *r;//simulations included in training
	int *v;//simulations held back for validation
	int nsimr,nsimv;
	int *Xset;
	bool normalizeinput;
	double **normalshift;
	double **normalgain;
	bool **weightdefined;
	bool **biasdefined;


	//black magic for modular fingerprints and activations
	class Activation ***activation;
	class Fingerprint ***fingerprints;
	typedef Fingerprint *(*FingerprintCreator)(PairRANN *);
	typedef Activation *(*ActivationCreator)(PairRANN *);
	typedef std::map<std::string,FingerprintCreator> FingerprintCreatorMap;
	typedef std::map<std::string,ActivationCreator> ActivationCreatorMap;
	FingerprintCreatorMap *fingerprint_map;
	ActivationCreatorMap *activation_map;
	Fingerprint * create_fingerprint(const char *);
	Activation * create_activation(const char *);

	//global variables
	int nelements;                // # of elements (distinct from LAMMPS atom types since multiple atom types can be mapped to one element)
	int nelementsp;				// nelements+1
	char **elements;              // names of elements
	char **elementsp;				// names of elements with "all" appended as the last "element"
	double *mass;                 // mass of each element
	double cutmax;				// max radial distance for neighbor lists
	int *map;                     // mapping from atom types to elements
	int *fingerprintcount;		// static variable used in initialization
	int *fingerprintlength;       // # of input neurons defined by fingerprints of each element.
	int *fingerprintperelement;   // # of fingerprints for each element
	bool doscreen;//screening is calculated if any defined fingerprint uses it
	bool allscreen;
	bool dospin;
	int res;//Resolution of function tables for cubic interpolation.
	double *screening_min;
	double *screening_max;
	int memguess;


	struct NNarchitecture{
	  int layers;
	  int *dimensions;//vector of length layers with entries for neurons per layer
	  int *dimensionsr;
	  double **Weights;
	  double **Biases;
	  int *activations;//unused
	  int maxlayer;//longest layer (for memory allocation)
	  int sumlayers;
	  int *startI;
	};
	NNarchitecture *net;//array of networks, 1 for each element.

	struct Simulation{
		bool forces;
		bool spins;
		int *id;
		double **x;
		double **f;
		double **s;
		double box[3][3];
		double origin[3];
		double **features;
		double **dfx;
		double **dfy;
		double **dfz;
		double **dsx;
		double **dsy;
		double **dsz;
		int *ilist,*numneigh,**firstneigh,*type,inum,gnum;
		double energy;
		double energy_weight;
		double force_weight;
		int startI;
	};
	Simulation *sims;
	void compute_jacobian(double *,double *,int *,int,int,NNarchitecture *);
	void qrsolve(double *,int,int,double*,double *);
	void chsolve(double *,int,double*,double *);
	void forward_pass(double *,int *,int,NNarchitecture *);
	void flatten_beta(NNarchitecture*,double*);//fill beta vector from net structure
	void unflatten_beta(NNarchitecture*,double*);//fill net structure from beta vector
	void copy_network(NNarchitecture*,NNarchitecture*);
	void normalize_data();
	void normalize_net(NNarchitecture*);
	void unnormalize_net(NNarchitecture*);

private:
	  template <typename T> static Fingerprint *fingerprint_creator(PairRANN *);
	  template <typename T> static Activation *activation_creator(PairRANN *);
};



}
#endif /* CALIBRATION_H_ */
