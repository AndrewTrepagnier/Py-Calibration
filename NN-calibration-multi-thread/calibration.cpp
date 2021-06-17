/*
 * PairRANN.cpp
 *
 *  Created on: Jul 27, 2020
 *      Author: kip
 */
#include "pair_rann.h"
#include "style_fingerprint.h"
#include "style_activation.h"
#include "omp.h"


using namespace LAMMPS_NS;

PairRANN::PairRANN(char *potential_file){
	cutmax = 0.0;
	nelementsp = -1;
	nelements = -1;
	net = NULL;
	fingerprintlength = NULL;
	mass = NULL;
	betalen = 0;
	doregularizer = false;
	normalizeinput = true;
	fingerprints = NULL;
	max_epochs = 1e7;
	regularizer = 0.0;
	res = 10000;
	fingerprintcount = 0;
	elementsp = NULL;
	elements = NULL;
	activation = NULL;
	tolerance = 1e-6;
	sims = NULL;
	doscreen = false;
	allscreen = true;
	dospin = false;
	map = NULL;//check this
	natoms = 0;
	nsims = 0;
	doforces = false;
	fingerprintperelement = NULL;
	validation = 0.0;
	potential_output_freq = 100;
	algorithm = new char [SHORTLINE];
	potential_input_file = new char [strlen(potential_file)+1];
	dump_directory = new char [SHORTLINE];
	log_file = new char [SHORTLINE];
	potential_output_file = new char [SHORTLINE];
	strncpy(this->potential_input_file,potential_file,strlen(potential_file)+1);
	char temp1[] = ".\0";
	char temp2[] = "calibration.log\0";
	char temp3[] = "potential_output.nn\0";
	strncpy(dump_directory,temp1,strlen(temp1)+1);
	strncpy(log_file,temp2,strlen(temp2)+1);
	strncpy(potential_output_file,temp3,strlen(temp3)+1);
	strncpy(algorithm,"LM_ch",strlen("LM_ch")+1);

	fingerprint_map = new FingerprintCreatorMap();

	#define FINGERPRINT_CLASS
	#define FingerprintStyle(key,Class) \
	  (*fingerprint_map)[#key] = &fingerprint_creator<Class>;
	#include "style_fingerprint.h"
	#undef FingerprintStyle
	#undef FINGERPRINT_CLASS

	activation_map = new ActivationCreatorMap();

	#define ACTIVATION_CLASS
	#define ActivationStyle(key,Class) \
	  (*activation_map)[#key] = &activation_creator<Class>;
	#include "style_activation.h"
	#undef ActivationStyle
	#undef ACTIVATION_CLASS
	srand(time(NULL));
	// srand(1234);
	// printf("!!! random seed is fixed to 1234.\n");
}

PairRANN::~PairRANN(){
	//clear memory
	delete [] algorithm;
	delete [] potential_input_file;
	delete [] dump_directory;
	delete [] log_file;
	delete [] potential_output_file;
	delete [] r;
	delete [] v;
	delete [] Xset;
	delete [] mass;
	for (int i=0;i<nsims;i++){
		for (int j=0;j<sims[i].inum;j++){
//			delete [] sims[i].x[j];
			if (doforces)delete [] sims[i].f[j];
			if (sims[i].spins)delete [] sims[i].s[j];
			delete [] sims[i].firstneigh[j];
			delete [] sims[i].features[j];
			if (doforces)delete [] sims[i].dfx[j];
			if (doforces)delete [] sims[i].dfy[j];
			if (doforces)delete [] sims[i].dfz[j];
		}
//		delete [] sims[i].x;
		if (doforces)delete [] sims[i].f;
		if (sims[i].spins)delete [] sims[i].s;
		if (doforces)delete [] sims[i].dfx;
		if (doforces)delete [] sims[i].dfy;
		if (doforces)delete [] sims[i].dfz;
		delete [] sims[i].firstneigh;
		delete [] sims[i].id;
		delete [] sims[i].features;
		delete [] sims[i].ilist;
		delete [] sims[i].numneigh;
		delete [] sims[i].type;
	}
	delete [] sims;
	for (int i=0;i<nelements;i++){delete [] elements[i];}
	delete [] elements;
	for (int i=0;i<nelementsp;i++){delete [] elementsp[i];}
	delete [] elementsp;
	for (int i=0;i<=nelements;i++){
		if (net[i].layers>0){
			for (int j=0;j<net[i].layers-1;j++){
				delete [] net[i].Weights[j];
				delete [] net[i].Biases[j];
				delete activation[i][j];
			}
			delete [] activation[i];
			delete [] net[i].dimensions;
			delete [] net[i].dimensionsr;
			delete [] net[i].Weights;
			delete [] net[i].Biases;
			delete [] net[i].startI;
		}
	}
	delete [] net;
	delete [] map;
	for (int i=0;i<nelementsp;i++){
		if (fingerprintlength[i]>0){
//			for (int j=0;j<fingerprintlength[i];j++){
//				delete fingerprints[i][j];
//			}
			delete [] fingerprints[i];
		}
	}
	delete [] fingerprints;
	delete [] activation;
	delete [] fingerprintcount;
	delete [] fingerprintperelement;
	delete [] fingerprintlength;
}

void PairRANN::setup(){

	int nthreads=1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();

	std::cout << std::endl;
	std::cout << "# Number Threads     : " << nthreads << std::endl;

	clock_t start = clock();

	read_potential_file();
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<fingerprintperelement[i];j++){
			  fingerprints[i][j]->allocate();
		}
	}
	read_dump_files();
	create_neighbor_lists();
	compute_fingerprints();
	if (normalizeinput){
		normalize_data();
	}
	separate_validation();

	clock_t end = clock();
	double time = (double) (end-start) / CLOCKS_PER_SEC;
	printf("finished setup(): %f seconds\n",time);
}

void PairRANN::run(){
	if (strcmp(algorithm,"LM_qr")==0){
		//slow but robust.
		levenburg_marquardt_qr();
	}
	else if (strcmp(algorithm,"LM_ch")==0){
		//faster. crashes if Jacobian has any columns of zeros.
		//usually will find exactly the same step for each iteration as qr.
		levenburg_marquardt_ch();
	}
	else if (strcmp(algorithm,"CG")==0){
		//faster iterations, but less accurate steps.
		//best parallelization
	}
	else {
		errorf("unrecognized algorithm");
	}
}


void PairRANN::finish(){

//	write_potential_file(true);
}

void PairRANN::read_potential_file(){
//	char str[MAXLINE];
	FILE *fid = fopen(potential_input_file,"r");
	if (fid==NULL){
		std::cout<<potential_input_file;
		errorf("Invalid potential file name");
	}
	std::cout<<"reading potential file\n";
	bool eof = false;
	bool comment;
	int nwords;
	char line[MAXLINE],line1[MAXLINE],*ptr;
	while (!eof){
		ptr = fgets(line,MAXLINE,fid);
		if (ptr==NULL){
			eof=true;
			//use default values for anything not defined
			check_parameters();
			if (check_potential()){errorf("potential is incomplete\n");}
			update_stack_size();
			return;
		}
		if ((ptr = strchr(line,'#'))) *ptr = '\0';//strip comments from end of lines
		if (count_words(line)==0){continue;}//skip comment line
		comment = true;
		while (comment==true){
			ptr = fgets(line1,MAXLINE,fid);
			if (ptr==NULL)errorf("Unexpected end of parameter file (keyword given with no value)");
			if ((ptr = strchr(line1,'#'))) *ptr = '\0';
			nwords = count_words(line1);
			if (nwords == 0) continue;
			comment = false;
		}
		line1[strlen(line1)-1] = '\0';//replace \n with \0
		nwords = count_words(line);
		char **words=new char *[nwords+1];
		nwords = 0;
		words[nwords++] = strtok(line,": ,\t_\n");
		while ((words[nwords++] = strtok(NULL,": ,\t_\n"))) continue;
		if (strcmp(words[0],"atomtypes")==0)read_atom_types(words,line1);
		else if (strcmp(words[0],"mass")==0)read_mass(words,line1);
		else if (strcmp(words[0],"fingerprintsperelement")==0)read_fpe(words,line1);
		else if (strcmp(words[0],"fingerprints")==0)read_fingerprints(words,nwords-1,line1);
		else if (strcmp(words[0],"fingerprintconstants")==0)read_fingerprint_constants(words,nwords-1,line1);
		else if (strcmp(words[0],"networklayers")==0)read_network_layers(words,line1);
		else if (strcmp(words[0],"layersize")==0)read_layer_size(words,line1);
		else if (strcmp(words[0],"weight")==0)read_weight(words,line1,fid);
		else if (strcmp(words[0],"bias")==0)read_bias(words,line1,fid);
		else if (strcmp(words[0],"activationfunctions")==0)read_activation_functions(words,line1);
		else if (strcmp(words[0],"screening")==0)read_screening(words,nwords-1,line1);
		else if (strcmp(words[0],"calibrationparameters")==0)read_parameters(words,line1);
		else errorf("Could not understand file syntax: unknown keyword");
		delete [] words;
	}
}

void PairRANN::read_parameters(char **words,char *line1){
	if (strcmp(words[1],"algorithm")==0){
		if (strlen(line1)>SHORTLINE){
			delete [] algorithm;
			algorithm = new char[strlen(line1)];
		}
		strncpy(algorithm,line1,SHORTLINE);
	}
	else if (strcmp(words[1],"dumpdirectory")==0){
		if (strlen(line1)>SHORTLINE){
			delete [] dump_directory;
			dump_directory = new char[strlen(line1)+1];
		}
		strncpy(dump_directory,line1,strlen(line1)+1);
	}
	else if (strcmp(words[1],"doforces")==0){
		int temp = strtol(line1,NULL,10);
		doforces = (temp>0);
	}
	else if (strcmp(words[1],"normalizeinput")==0){
		int temp = strtol(line1,NULL,10);
		normalizeinput = (temp>0);
	}
	else if (strcmp(words[1],"tolerance")==0){
		tolerance = strtod(line1,NULL);
	}
	else if (strcmp(words[1],"regularizer")==0){
		regularizer = strtod(line1,NULL);
		doregularizer = true;
	}
	else if (strcmp(words[1],"logfile")==0){
		if (strlen(line1)>SHORTLINE){
			delete [] log_file;
			log_file = new char [strlen(line1)+1];
		}
		strncpy(log_file,line1,strlen(line1)+1);
	}
	else if (strcmp(words[1],"potentialoutputfreq")==0){
		potential_output_freq = strtol(line1,NULL,10);
	}
	else if (strcmp(words[1],"potentialoutputfile")==0){
		if (strlen(line1)>SHORTLINE){
			delete [] potential_output_file;
			potential_output_file = new char [strlen(line1)+1];
		}
		strncpy(potential_output_file,line1,strlen(line1)+1);
	}
	else if (strcmp(words[1],"maxepochs")==0){
		max_epochs = strtol(line1,NULL,10);
	}
	else if (strcmp(words[1],"dimsreserved")==0){
		int i;
		for (i=0;i<nelements;i++){
			if (strcmp(words[2],elements[i])==0){
				if (net[i].layers==0)errorf("networklayers for each atom type must be defined before the corresponding layer sizes.");
				int j = strtol(words[3],NULL,10);
				net[i].dimensionsr[j]= strtol(line1,NULL,10);
				return;
			}
		}
		errorf("dimsreserved element not found in atom types");
	}
	else if (strcmp(words[1],"validation")==0){
		validation = strtod(line1,NULL);
	}
	else {
		char str[MAXLINE];
		sprintf(str,"unrecognized keyword in parameter file: %s\n",words[1]);
		errorf(str);
	}
}

void PairRANN::create_random_weights(int rows,int columns,int itype,int layer){
	net[itype].Weights[layer] = new double [rows*columns];
	double r;
	for (int i=0;i<rows;i++){
		for (int j=0;j<columns;j++){
			r = (double)rand()/RAND_MAX*2-1;
			net[itype].Weights[layer][j*rows+i] = r;
		}
	}
}

void PairRANN::create_random_biases(int rows,int itype, int layer){
	net[itype].Biases[layer] = new double [rows];
	double r;
	for (int i=0;i<rows;i++){
		r = (double) rand()/RAND_MAX*2-1;
		net[itype].Biases[layer][i] = r;
	}
}

void PairRANN::allocate(char **elementword)
{
	int i,n;
	cutmax = 0;
	nelementsp=nelements+1;
	//initialize arrays
	elements = new char *[nelements];
	elementsp = new char *[nelementsp];//elements + 'all'
	map = new int[nelementsp];
	mass = new double[nelements];
	net = new NNarchitecture[nelementsp];
	betalen_v = new int[nelementsp];
	screening_min = new double [nelements*nelements*nelements];
	screening_max = new double [nelements*nelements*nelements];
	for (i=0;i<nelements;i++){
		for (int j =0;j<nelements;j++){
			for (int k=0;k<nelements;k++){
				screening_min[i*nelements*nelements+j*nelements+k] = 0.8;//default values. Custom values may be read from potential file later.
				screening_max[i*nelements*nelements+j*nelements+k] = 2.8;//default values. Custom values may be read from potential file later.
			}
		}
	}
	weightdefined = new bool*[nelementsp];
	biasdefined = new bool *[nelementsp];
	activation = new Activation**[nelementsp];
	fingerprints = new Fingerprint**[nelementsp];
	fingerprintlength = new int[nelementsp];
	fingerprintperelement = new int [nelementsp];
	fingerprintcount = new int[nelementsp];

	for (i=0;i<=nelements;i++){
		n = strlen(elementword[i])+1;
		fingerprintlength[i]=0;
		fingerprintperelement[i] = -1;
		fingerprintcount[i] = 0;
		map[i] = i;
		if (i<nelements){
			mass[i]=-1.0;
			elements[i]= new char[n];
			strncpy(elements[i],elementword[i],n);
		}
		elementsp[i] = new char[n];
		strncpy(elementsp[i],elementword[i],n);
		net[i].layers = 0;
		net[i].dimensions = new int[1];
		net[i].dimensions[0]=0;
		net[i].dimensionsr = new int[1];
		net[i].dimensionsr[0]=0;
	}

}

//TO DO:Improve error messages in here
//Called after finishing reading the potential file to make sure it is complete. True is bad.
//also allocates maxlayer and fingerprintlength and new weights and biases if ones were not provided.
bool PairRANN::check_potential(){
  int i,j,k,l;
  char str[MAXLINE];
  if (nelements==-1){return true;}
  for (i=0;i<=nelements;i++){
	  if (i<nelements){
		  if (mass[i]<0)return true;//uninitialized mass
	  }
	  if (net[i].layers==0)continue;//no definitions for this starting element, not considered an error.
	  net[i].maxlayer=0;
	  net[i].sumlayers=0;
	  net[i].startI = new int [net[i].layers];
	  betalen_v[i] = 0;
	  if (i>0){
		  betalen_v[i]+=betalen_v[i-1];
	  }
	  for (j=0;j<net[i].layers;j++){
		  net[i].startI[j] = net[i].sumlayers;
		  net[i].sumlayers+=net[i].dimensions[j];
		  if (net[i].dimensions[j]<=0)return true;//incomplete network definition
		  if (net[i].dimensionsr[j]<0 || net[i].dimensionsr[j]>net[i].dimensions[j])return true;
		  if (net[i].dimensions[j]>net[i].maxlayer)net[i].maxlayer = net[i].dimensions[j];
	  }
	  if (net[i].dimensions[net[i].layers-1]!=1)return true;//output layer must have single neuron (the energy)
	  for (j=0;j<net[i].layers-1;j++){
//		  betalen +=(net[i].dimensions[j]-net[i].dimensionsr[j])*(net[i].dimensions[j+1]-net[i].dimensionsr[j+1]);
		  betalen +=(net[i].dimensions[j]-net[i].dimensionsr[j])*(net[i].dimensions[j+1]-0);
		  //betalen +=(net[i].dimensions[j+1]-net[i].dimensionsr[j+1]);
		  if (net[i].dimensionsr[j]==0)betalen+=net[i].dimensions[j+1];
//		  betalen_v[i] +=(net[i].dimensions[j]-net[i].dimensionsr[j])*(net[i].dimensions[j+1]-net[i].dimensionsr[j+1]);
		  betalen_v[i] +=(net[i].dimensions[j]-net[i].dimensionsr[j])*(net[i].dimensions[j+1]-0);
		  //betalen_v[i] +=(net[i].dimensions[j+1]-net[i].dimensionsr[j+1]);
		  if (net[i].dimensionsr[j]==0)betalen_v[i]+=net[i].dimensions[j+1];

		  if (!weightdefined[i][j])create_random_weights(net[i].dimensions[j],net[i].dimensions[j+1],i,j);//undefined weights
		  if (!biasdefined[i][j])create_random_biases(net[i].dimensions[j+1],i,j);//undefined biases
		  if (activation[i][j]->empty)return true;//undefined activations
	  }
	  for (j=0;j<fingerprintperelement[i];j++){
		  if (fingerprints[i][j]->fullydefined==false)return true;
		  fingerprints[i][j]->startingneuron = fingerprintlength[i];
		  fingerprintlength[i] +=fingerprints[i][j]->get_length();
		  if (fingerprints[i][j]->rc>cutmax){cutmax = fingerprints[i][j]->rc;}
		  if (fingerprints[i][j]->spin==true){dospin=true;}
	  }
	  if (net[i].dimensions[0]!=fingerprintlength[i])return true;
  }
  return false;//everything looks good
}

void PairRANN::update_stack_size(){
	//get very rough guess of memory usage
	int jlen = nsims;
	if (doregularizer){
		jlen+=betalen-1;
	}
	if (doforces){
		jlen+=natoms*3;
	}
	//neighborlist memory use:
	memguess = 0;
	for (int i=0;i<nelementsp;i++){
		memguess+=8*net[i].dimensions[0]*20*3;
	}
	memguess+=8*20*12;
	memguess+=8*20*20*3;
	//separate validation memory use:
	memguess+=nsims*8*2;
	//levenburg marquardt ch memory use:
	memguess+=8*jlen*betalen*2;
	memguess+=8*betalen*betalen;
	memguess+=8*jlen*4;
	memguess+=8*betalen*4;
	//chsolve memory use:
	memguess+=8*betalen*betalen;
	//generous buffer:
	memguess *= 16;
	const rlim_t kStackSize = memguess;
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur += kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

bool PairRANN::check_parameters(){
	if (strcmp(algorithm,"LM_qr")!=0 && strcmp(algorithm,"LM_ch")!=0)return true;//add others later maybe
	if (tolerance==0.0 && max_epochs == 0)return true;
	if (tolerance<0.0 || max_epochs < 0 || regularizer < 0.0 || potential_output_freq < 0)return true;
	return false;//everything looks good
}

//part of setup. Do not optimize:
void PairRANN::read_dump_files(){
	DIR *folder;
//	char str[MAXLINE];
	struct dirent *entry;
	int file = 0;
	char line[MAXLINE],*ptr;
	char **words;
	int nwords,nwords1,sets;
	folder = opendir(dump_directory);

	if(folder == NULL)
	{
		errorf("unable to open dump directory");
	}
	std::cout<<"reading dump files\n";
	int nsims = 0;
	int nsets = 0;
	//count files
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		nsets++;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	this->nsets = nsets;
	Xset=new int[nsets];
	int count=0;
	//count snapshots per file
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		ptr = fgets(line,MAXLINE,fid);
		nwords = 0;
		words = new char* [strlen(line)];
		words[nwords++] = strtok(line," ,\t\n");
		while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
		nwords--;
		if (nwords!=5){errorf("dumpfile must contain 2nd line with timestep, energy, energy_weight, force_weight, snapshots\n");}
		sets = strtol(words[4],NULL,10);
		delete [] words;
		nsims+=sets;
		Xset[count++]=sets;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	sims = new Simulation[nsims];
	this->nsims = nsims;
	sims[0].startI=0;
	//read dump files
	while((entry=readdir(folder))){
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		while (ptr!=NULL){
			if (strstr(line,"ITEM: TIMESTEP")==NULL)errorf("invalid dump file line 1");
//			if (file==396){
//				std::cout<<entry->d_name;
//			}
			ptr = fgets(line,MAXLINE,fid);//timestep
			nwords = 0;
			char *words1[strlen(line)];
			words1[nwords++] = strtok(line," ,\t");
			while ((words1[nwords++] = strtok(NULL," ,\t"))) continue;
			nwords--;
			if (nwords!=5)errorf("error: dump file line 2 must contain 5 entries: timestep, energy, energy_weight, force_weight, snapshots");
			sims[file].energy = strtod(words1[1],NULL);
			sims[file].energy_weight = strtod(words1[2],NULL);
			sims[file].force_weight = strtod(words1[3],NULL);
			ptr = fgets(line,MAXLINE,fid);//ITEM: NUMBER OF ATOMS
			if (strstr(line,"ITEM: NUMBER OF ATOMS")==NULL)errorf("invalid dump file line 3");
			ptr = fgets(line,MAXLINE,fid);//natoms
			int natoms = strtol(line,NULL,10);
			if (file>0){sims[file].startI=sims[file-1].startI+natoms*3;}
			this->natoms+=natoms;
			sims[file].energy_weight /=natoms;
			//sims[file].force_weight /=natoms;
			ptr = fgets(line,MAXLINE,fid);//ITEM: BOX BOUNDS xy xz yz pp pp pp
			if (strstr(line,"ITEM: BOX BOUNDS")==NULL)errorf("invalid dump file line 5");
			double box[3][3];
			double origin[3];
			bool cols[12];
			for (int i= 0;i<11;i++){
				cols[i]=false;
			}
			box[0][1] = box[0][2] = box[1][2] = 0.0;
			for (int i = 0;i<3;i++){
				ptr = fgets(line,MAXLINE,fid);//box line
				char *words[4];
				nwords = 0;
				words[nwords++] = strtok(line," ,\t\n");
				while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
				nwords--;
				if (nwords!=3 && nwords!=2){errorf("invalid dump box definition");}
				origin[i] = strtod(words[0],NULL);
				box[i][i] = strtod(words[1],NULL);
				if (nwords==3){
					if (i==0){
						box[0][1]=strtod(words[2],NULL);
						if (box[0][1]>0){box[0][0]-=box[0][1];}
						else origin[0] -= box[0][1];
					}
					else if (i==1){
						box[0][2]=strtod(words[2],NULL);
						if (box[0][2]>0){box[0][0]-=box[0][2];}
						else origin[0] -= box[0][2];
					}
					else{
						box[1][2]=strtod(words[2],NULL);
						if (box[1][2]>0)box[1][1]-=box[1][2];
						else origin[1] -=box[1][2];
					}
				}
			}
			for (int i=0;i<3;i++)box[i][i]-=origin[i];
			box[1][0]=box[2][0]=box[2][1]=0.0;
			ptr = fgets(line,MAXLINE,fid);//ITEM: ATOMS id type x y z c_energy fx fy fz sx sy sz
			nwords = 0;
			char *words[count_words(line)+1];
			words[nwords++] = strtok(line," ,\t\n");
			while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			int colid = -1;
			int columnmap[10];
			for (int i=0;i<nwords-2;i++){columnmap[i]=-1;}
			for (int i=2;i<nwords;i++){
				if (strcmp(words[i],"type")==0){colid = 0;}
				else if (strcmp(words[i],"x")==0){colid=1;}
				else if (strcmp(words[i],"y")==0){colid=2;}
				else if (strcmp(words[i],"z")==0){colid=3;}
				//else if (strcmp(words[i],"c_energy")==0){colid=4;}
				else if (strcmp(words[i],"fx")==0){colid=4;}
				else if (strcmp(words[i],"fy")==0){colid=5;}
				else if (strcmp(words[i],"fz")==0){colid=6;}
				else if (strcmp(words[i],"sx")==0){colid=7;}
				else if (strcmp(words[i],"sy")==0){colid=8;}
				else if (strcmp(words[i],"sz")==0){colid=9;}
				else {continue;}
				cols[colid] = true;
				if (colid!=-1){columnmap[colid]=i-2;}
			}
			for (int i=0;i<4;i++){
				if (!cols[i]){errorf("dump file must include type, x, y, and z data columns (other recognized keywords are fx, fy, fz, sx, sy, sz)");}
			}
			bool doforce = false;
			bool dospin = false;
			sims[file].inum = natoms;
			sims[file].ilist = new int [natoms];
			sims[file].type = new int [natoms];
			sims[file].x= new double *[natoms];
			for (int i=0;i<3;i++){
				for (int j=0;j<3;j++)sims[file].box[i][j]=box[i][j];
				sims[file].origin[i]=origin[i];
			}
			//sims[file].energy = new double [natoms];
			for (int i=0;i<natoms;i++){
				sims[file].x[i]=new double [3];
			}
			//if force calibration is on
			if (doforces){
				sims[file].f = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].f[i] = new double [3];
				}
			}
			//if forces are given in dump file
			if (cols[4] && cols[5] && cols[6] && doforces){
				doforce = true;
			}
			if (cols[7] && cols[8] && cols[9]){
				dospin = true;
				sims[file].s = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].s[i] = new double [3];
				}
			}
			else if (this->dospin){
				errorf("spin vectors must be defined for all input simulations when magnetic fingerprints are used\n");
			}
			for (int i=0;i<natoms;i++){
				ptr = fgets(line,MAXLINE,fid);
				char *words2[count_words(line)+1];
				nwords1 = 0;
				words2[nwords1++] = strtok(line," ,\t");
				while ((words2[nwords1++] = strtok(NULL," ,\t"))) continue;
				nwords1--;
				if (nwords1!=nwords-2){errorf("incorrect number of data columns in dump file.");}
				sims[file].ilist[i]=i;//ignore any id mapping in the dump file, just id them based on line number.
				sims[file].type[i]=strtol(words2[columnmap[0]],NULL,10)-1;//lammps type counting starts at 1 instead of 0
				sims[file].x[i][0]=strtod(words2[columnmap[1]],NULL);
				sims[file].x[i][1]=strtod(words2[columnmap[2]],NULL);
				sims[file].x[i][2]=strtod(words2[columnmap[3]],NULL);
				//sims[file].energy[i]=strtod(words[columnmap[4]],NULL);
				if (doforce){
					sims[file].f[i][0]=strtod(words2[columnmap[4]],NULL);
					sims[file].f[i][1]=strtod(words2[columnmap[5]],NULL);
					sims[file].f[i][2]=strtod(words2[columnmap[6]],NULL);
				}
				//if force calibration is on, but forces are not given in file, assume they are zero.
				else if (doforces){
					sims[file].f[i][0]=0.0;
					sims[file].f[i][1]=0.0;
					sims[file].f[i][2]=0.0;
				}
				if (dospin){
					sims[file].s[i][0]=strtod(words2[columnmap[7]],NULL);
					sims[file].s[i][1]=strtod(words2[columnmap[8]],NULL);
					sims[file].s[i][2]=strtod(words2[columnmap[9]],NULL);
				}
				sims[file].spins = dospin;
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
			file++;
			if (file>nsims){errorf("Too many dump files found. Nsims is incorrect.\n");}
		}
		fclose(fid);
	}

	closedir(folder);
	sprintf(line,"imported %d atoms, %d simulations\n",natoms,nsims);
	std::cout<<line;
}

//part of setup. Do not optimize:
void PairRANN::create_neighbor_lists(){
	//brute force search technique rather than tree search because we only do it once and most simulations are small.
	//I did optimize for low memory footprint by only adding ghost neighbors
	//within cutoff distance of the box
	int i,ix,iy,iz,j,k;
//	char str[MAXLINE];
	double buffer = 0.01;//over-generous compensation for roundoff error
	std::cout<<"building neighbor lists\n";
	for (i=0;i<nsims;i++){
		double box[3][3];
		for (ix=0;ix<3;ix++){
			for (iy=0;iy<3;iy++)box[ix][iy]=sims[i].box[ix][iy];
		}
		double *origin = sims[i].origin;
		int natoms = sims[i].inum;
		int xb = floor(cutmax/box[0][0]+1);
		int yb = floor(cutmax/box[1][1]+1);
		int zb = floor(cutmax/box[2][2]+1);
		int buffsize = natoms*(xb*2+1)*(yb*2+1)*(zb*2+1);
		double x[buffsize][3];
		int type[buffsize];
		int id[buffsize];
		int count = 0;

		//force all atoms to be inside the box:
		double xtemp[3];
		double xp[3];
		double boxt[9];
		for (j=0;j<3;j++){
			for (k=0;k<3;k++){
				boxt[j*3+k]=box[j][k];
			}
		}
		for (j=0;j<natoms;j++){
			for (k=0;k<3;k++){
				xp[k] = sims[i].x[j][k]-origin[k];
			}
			qrsolve(boxt,3,3,xp,xtemp);//convert coordinates from Cartesian to box basis
			for (k=0;k<3;k++){
				xtemp[k]-=floor(xtemp[k]);//if atom is outside box find periodic replica in box
			}
			for (k=0;k<3;k++){
				sims[i].x[j][k] = 0.0;
				for (int l=0;l<3;l++){
					sims[i].x[j][k]+=box[k][l]*xtemp[l];//convert back to Cartesian
				}
				sims[i].x[j][k]+=origin[k];
			}
		}

		//calculate box face normal directions and plane intersections
		double xpx,xpy,xpz,ypx,ypy,ypz,zpx,zpy,zpz;
		zpx = 0;zpy=0;zpz =1;
		double ym,xm;
		ym = sqrt(box[1][2]*box[1][2]+box[2][2]*box[2][2]);
		xm = sqrt(box[1][1]*box[2][2]*box[1][1]*box[2][2]+box[0][1]*box[0][1]*box[2][2]*box[2][2]+(box[0][1]*box[1][2]-box[0][2]*box[1][1])*(box[0][1]*box[1][2]-box[0][2]*box[1][1]));
		//unit vectors normal to box faces:
		ypx = 0;
		ypy = box[2][2]/ym;
		ypz = -box[1][2]/ym;
		xpx = box[1][1]*box[2][2]/xm;
		xpy = -box[0][1]*box[2][2]/xm;
		xpz = (box[0][1]*box[1][2]-box[0][2]*box[1][1])/xm;
		double fxn,fxp,fyn,fyp,fzn,fzp;
		//minimum distances from origin to planes aligned with box faces:
		fxn = origin[0]*xpx+origin[1]*xpy+origin[2]*xpz;
		fyn = origin[0]*ypx+origin[1]*ypy+origin[2]*ypz;
		fzn = origin[0]*zpx+origin[1]*zpy+origin[2]*zpz;
		fxp = (origin[0]+box[0][0])*xpx+(origin[1]+box[1][0])*xpy+(origin[2]+box[2][0])*xpz;
		fyp = (origin[0]+box[0][1])*ypx+(origin[1]+box[1][1])*ypy+(origin[2]+box[2][1])*ypz;
		fzp = (origin[0]+box[0][2])*zpx+(origin[1]+box[1][2])*zpy+(origin[2]+box[2][2])*zpz;
		//fill buffered atom list
		double px,py,pz;
		double xe,ye,ze;
		for (j=0;j<natoms;j++){
			x[count][0] = sims[i].x[j][0];
			x[count][1] = sims[i].x[j][1];
			x[count][2] = sims[i].x[j][2];
			type[count] = sims[i].type[j];
			id[count] = j;
			count++;
		}

		//add ghost atoms outside periodic boundaries:
		for (ix=-xb;ix<=xb;ix++){
			for (iy=-yb;iy<=yb;iy++){
				for (iz=-zb;iz<=zb;iz++){
					if (ix==0 && iy == 0 && iz == 0)continue;
					for (j=0;j<natoms;j++){
						xe = ix*box[0][0]+iy*box[0][1]+iz*box[0][2]+sims[i].x[j][0];
						ye = iy*box[1][1]+iz*box[1][2]+sims[i].x[j][1];
						ze = iz*box[2][2]+sims[i].x[j][2];
						px = xe*xpx+ye*xpy+ze*xpz;
						py = xe*ypx+ye*ypy+ze*ypz;
						pz = xe*zpx+ye*zpy+ze*zpz;
						//include atoms if their distance from the box face is less than cutmax
						if (px>cutmax+fxp+buffer || px<fxn-cutmax-buffer){continue;}
						if (py>cutmax+fyp+buffer || py<fyn-cutmax-buffer){continue;}
						if (pz>cutmax+fzp+buffer || pz<fzn-cutmax-buffer){continue;}
						x[count][0] = xe;
						x[count][1] = ye;
						x[count][2] = ze;
						type[count] = sims[i].type[j];
						id[count] = j;
						count++;
						if (count>buffsize){errorf("neighbor overflow!\n");}
					}
				}
			}
		}

		//update stored lists
		buffsize = count;
		for (j=0;j<natoms;j++){
			delete [] sims[i].x[j];
		}
		delete [] sims[i].x;
		delete [] sims[i].type;
		delete [] sims[i].ilist;
		sims[i].type = new int [buffsize];
		sims[i].x = new double *[buffsize];
		sims[i].id = new int [buffsize];
		sims[i].ilist = new int [buffsize];
		for (j=0;j<buffsize;j++){
			sims[i].x[j] = new double [3];
			for (k=0;k<3;k++){
				sims[i].x[j][k] = x[j][k];
			}
			sims[i].type[j] = type[j];
			sims[i].id[j] = id[j];
			sims[i].ilist[j] = j;
//			delete [] x[j];
		}
//		delete [] x;
//		delete [] type;
//		delete [] id;
		sims[i].inum = natoms;
		sims[i].gnum = buffsize-natoms;
		sims[i].numneigh = new int[natoms];
		sims[i].firstneigh = new int*[natoms];
		//do double count, slow, but enables getting the exact size of the neighbor list before filling it.
		for (j=0;j<natoms;j++){
			sims[i].numneigh[j]=0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].numneigh[j]++;
				}
			}
			sims[i].firstneigh[j] = new int[sims[i].numneigh[j]];
			count = 0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].firstneigh[j][count] = k;
//					if (i==396 && j==0){
//						sprintf(str,"%.16f %.16f %.16f\n",xtmp,ytmp,ztmp);
//						std::cout<<str;
//					}
					count++;
				}
			}
		}
//		for (k=0;k<buffsize;k++){
//			sprintf(str,"%d %d %f %f %f\n",k+1,sims[i].type[k]+1,sims[i].x[k][0],sims[i].x[k][1],sims[i].x[k][2]);
//			std::cout<<str;
//		}
//		sprintf(str,"%d\n",buffsize);
//		std::cout<<str;
	}
}

//part of setup. Do not optimize:
//TO DO: fix stack size problem
void PairRANN::compute_fingerprints(){
//	char str[MAXLINE];
	std::cout<<"computing fingerprints\n";
	int i,ii,itype,f,jnum,len,j,nn;
	for (nn=0;nn<nsims;nn++){
		sims[nn].features = new double *[sims[nn].inum];
		if (doforces){
			sims[nn].dfx = new double *[sims[nn].inum];
			sims[nn].dfy = new double *[sims[nn].inum];
			sims[nn].dfz = new double *[sims[nn].inum];
		}
		  for (ii=0;ii<sims[nn].inum;ii++){
			  i = sims[nn].ilist[ii];
			  itype = map[sims[nn].type[i]];
			  f = net[itype].dimensions[0];
			  jnum = sims[nn].numneigh[i];
			  double xn[jnum];
			  double yn[jnum];
			  double zn[jnum];
			  int tn[jnum];
			  int jl[jnum];
			  cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn);
			  double features [f];
			  double dfeaturesx[f*jnum];
			  double dfeaturesy[f*jnum];
			  double dfeaturesz[f*jnum];
			  for (j=0;j<f;j++){
				  features[j]=0;
			  }
			  for (j=0;j<f*jnum;j++){
				  dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
			  }
			  //screening is calculated once for all atoms if any fingerprint uses it.
			  double Sik[jnum];
			  double dSikx[jnum];
			  double dSiky[jnum];
			  double dSikz[jnum];
			  //TO D0: add check against stack size
			  double dSijkx[jnum*jnum];
			  double dSijky[jnum*jnum];
			  double dSijkz[jnum*jnum];
			  bool Bij[jnum];
			  double sx[jnum*f];
			  double sy[jnum*f];
			  double sz[jnum*f];
			  if (doscreen){
					screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
			  }
			  if (allscreen){
				  screen_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,Bij,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz);
			  }
//			  sims[nn].numneigh[sims[nn].ilist[ii]] = jnum-1;
//			  for (j=0;j<jnum-1;j++){
//				  sims[nn].firstneigh[ii][j] = jl[j];
//			  }
			  //do fingerprints for atom type
			  len = fingerprintperelement[itype];
//			  sprintf(str,"%d %d\n",itype,len);
//			  std::cout<<str;
			  for (j=0;j<len;j++){
				  	  	   if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			  }
			  itype = nelements;
			  //do fingerprints for type "all"
			  len = fingerprintperelement[itype];
			  for (j=0;j<len;j++){
				  	  	   if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			  }
			  //copy features from stack to heap
			  sims[nn].features[ii] = new double [f];
//			  sprintf(str,"sim %d atom %d\n",nn,ii);
//			  std::cout<<str;
			  for (j=0;j<f;j++){
				  sims[nn].features[ii][j] = features[j];
			  }
//			  if (nn == 1){
//				  for (j=0;j<f;j++){
//				  sprintf(str,"%.16f  ",features[j]);
//				  std::cout<<str;
//				  }
//				  sprintf(str,"%d %d\n",f,map[sims[nn].type[i]]);
//				  std::cout<<str;
//			  }
			  if (doforces){
				  sims[nn].dfx[ii] = new double[f*jnum];
				  sims[nn].dfy[ii] = new double[f*jnum];
				  sims[nn].dfz[ii] = new double[f*jnum];
				  for (j=0;j<f*jnum;j++){
					  sims[nn].dfx[ii][j]=dfeaturesx[j];
					  sims[nn].dfy[ii][j]=dfeaturesy[j];
					  sims[nn].dfz[ii][j]=dfeaturesz[j];
				  }
				  if (dospin){
					  sims[nn].dsx[ii] = new double[f*jnum];
					  sims[nn].dsy[ii] = new double[f*jnum];
					  sims[nn].dsz[ii] = new double[f*jnum];
					  for (j=0;j<f*jnum;j++){
						  sims[nn].dsx[ii][j] = sx[j];
						  sims[nn].dsy[ii][j] = sy[j];
						  sims[nn].dsz[ii][j] = sz[j];
					  }
				  }
			  }
		  }
		  for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
			  delete [] sims[nn].x[ii];
		  }
		  delete [] sims[nn].x;
	}
}

void PairRANN::normalize_data(){
	int i,n,ii,j,itype;
	int natoms[nelementsp];
	normalgain = new double *[nelementsp];
	normalshift = new double *[nelementsp];
	//initialize
	for (i=0;i<nelementsp;i++){
		normalgain[i] = new double [net[i].dimensions[0]];
		normalshift[i] = new double [net[i].dimensions[0]];
		for (j=0;j<net[i].dimensions[0];j++){
			normalgain[i][j]=0;
			normalshift[i][j]=0;
		}
		natoms[i] = 0;
	}
	//get mean value of each 1st layer neuron input
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			natoms[itype]++;
			if (net[itype].layers!=0){
				if (net[itype].dimensionsr[0]==0){
					for (j=0;j<net[itype].dimensions[0];j++){
						normalshift[itype][j]+=sims[n].features[ii][j];
					}
				}
				else {//fixes incompatibility between fixed biases and normalized inputs
					for (j=0;j<net[itype].dimensionsr[0];j++){
						normalshift[itype][j]+=sims[n].features[ii][j];
					}
				}
			}
			itype = nelements;
			natoms[itype]++;
			if (net[itype].layers!=0){
				if (net[itype].dimensionsr[0]==0){
					for (j=0;j<net[itype].dimensions[0];j++){
						normalshift[itype][j]+=sims[n].features[ii][j];
					}
				}
				else {
					for (j=0;j<net[itype].dimensionsr[0];j++){
						normalshift[itype][j]+=sims[n].features[ii][j];
					}
				}
			}
		}
	}
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net[i].dimensions[0];j++){
			normalshift[i][j]/=natoms[i];
		}
	}
	//get standard deviation
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalgain[itype][j]+=(sims[n].features[ii][j]-normalshift[itype][j])*(sims[n].features[ii][j]-normalshift[itype][j]);
				}
			}
			itype = nelements;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalshift[itype][j]+=(sims[n].features[ii][j]-normalshift[itype][j])*(sims[n].features[ii][j]-normalshift[itype][j]);
				}
			}
		}
	}
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net[i].dimensions[0];j++){
			normalgain[i][j]=sqrt(normalgain[i][j]/natoms[i]);
		}
	}
	//shift input to mean=0, std = 1
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					if (normalgain[itype][j]>0){
						sims[n].features[ii][j] -= normalshift[itype][j];
						sims[n].features[ii][j] /= normalgain[itype][j];
					}
				}
			}
			itype = nelements;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					if (normalgain[itype][j]>0){
						sims[n].features[ii][j] -= normalshift[itype][j];
						sims[n].features[ii][j] /= normalgain[itype][j];
					}
				}
			}
		}
	}
	NNarchitecture *net_new = new NNarchitecture[nelementsp];
	normalize_net(net_new);
	copy_network(net_new,net);
	delete [] net_new;
}

void PairRANN::unnormalize_net(NNarchitecture *net_out){
	int i,j,k;
	double temp;
	copy_network(net,net_out);
	for (i=0;i<nelementsp;i++){
		if (net[i].layers>0){
			for (j=0;j<net[i].dimensions[1];j++){
				temp = 0.0;
				for (k=0;k<net[i].dimensions[0];k++){
					if (normalgain[i][k]>0){
						net_out[i].Weights[0][j*net[i].dimensions[0]+k]/=normalgain[i][k];
						temp+=net_out[i].Weights[0][j*net[i].dimensions[0]+k]*normalshift[i][k];
					}
				}
				net_out[i].Biases[0][j]-=temp;
			}
		}
	}
}

void PairRANN::normalize_net(NNarchitecture *net_out){
	int i,j,k;
	double temp;
	copy_network(net,net_out);
	for (i=0;i<nelementsp;i++){
		if (net[i].layers>0){
			for (j=0;j<net[i].dimensions[1];j++){
				temp = 0.0;
				for (k=0;k<net[i].dimensions[0];k++){
					if (normalgain[i][k]>0){
						temp+=net_out[i].Weights[0][j*net[i].dimensions[0]+k]*normalshift[i][k];
						if (weightdefined[i][0])net_out[i].Weights[0][j*net[i].dimensions[0]+k]*=normalgain[i][k];
					}
				}
				if (biasdefined[i][0])net_out[i].Biases[0][j]+=temp;
			}
		}
	}
}

void PairRANN::separate_validation(){
	int n1,n2,i,vnum,len,startI,endI,j,t,k;
	char str[MAXLINE];
	int Iv[nsims];
	int Ir[nsims];
	bool w;
	n1=n2=0;
	sprintf(str,"finishing setup\n");
	std::cout<<str;
	for (i=0;i<nsims;i++)Iv[i]=-1;
	for (i=0;i<nsets;i++){
		startI=0;
		for (j=0;j<i;j++)startI+=Xset[j];
		endI = startI+Xset[i];
		len = Xset[i];
		vnum = rand();
		if (vnum<floor(RAND_MAX*validation)){
			vnum = 1;
		}
		else{
			vnum = 0;
		}
		vnum+=floor(len*validation);
		while (vnum>0){
			w = true;
			t = floor(rand() % len)+startI;
			for (j=0;j<n1;j++){
				if (t==Iv[j]){
					w = false;
					break;
				}
			}
			if (w){
				Iv[n1]=t;
				vnum--;
				n1++;
			}
		}
		for (j=startI;j<endI;j++){
			w = true;
			for (k=0;k<n1;k++){
				if (j==Iv[k]){
					w = false;
					break;
				}
			}
			if (w){
				Ir[n2]=j;
				n2++;
			}
		}
	}
	nsimr = n2;
	nsimv = n1;
	r = new int [n2];
	v = new int [n1];
	natomsr = 0;
	natomsv = 0;
	for (i=0;i<n1;i++){
		v[i]=Iv[i];
		natomsv += sims[v[i]].inum;
	}
	for (i=0;i<n2;i++){
		r[i]=Ir[i];
		natomsr += sims[r[i]].inum;
	}
	sprintf(str,"assigning %d simulations (%d atoms) for validation, %d simulations (%d atoms) for fitting\n",nsimv,natomsv,nsimr,natomsr);
	std::cout<<str;
}

void PairRANN::copy_network(NNarchitecture *net_old,NNarchitecture *net_new){
	int i,j,k;
	for (i=0;i<nelementsp;i++){
		net_new[i].layers = net_old[i].layers;
		if (net_new[i].layers>0){
			net_new[i].maxlayer = net_old[i].maxlayer;
			net_new[i].sumlayers=net_old[i].sumlayers;
			net_new[i].dimensions = new int [net_new[i].layers];
			net_new[i].dimensionsr = new int [net_new[i].layers];
			net_new[i].startI = new int [net_new[i].layers];
			net_new[i].Weights = new double*[net_new[i].layers-1];
			net_new[i].Biases = new double*[net_new[i].layers-1];
			for (j=0;j<net_old[i].layers;j++){
				net_new[i].dimensions[j]=net_old[i].dimensions[j];
				net_new[i].dimensionsr[j]=net_old[i].dimensionsr[j];
				net_new[i].startI[j]=net_old[i].startI[j];
				if (j>0){
					net_new[i].Weights[j-1] = new double [net_new[i].dimensions[j]*net_new[i].dimensions[j-1]];
					net_new[i].Biases[j-1] = new double [net_new[i].dimensions[j]];
					for (k=0;k<net_new[i].dimensions[j]*net_new[i].dimensions[j-1];k++){
						net_new[i].Weights[j-1][k]=net_old[i].Weights[j-1][k];
					}
					for (k=0;k<net_new[i].dimensions[j];k++){
						net_new[i].Biases[j-1][k] = net_old[i].Biases[j-1][k];
					}
				}
			}
		}
	}
}

//top level run function, calls compute_jacobian and qrsolve. Cannot be parallelized.
void PairRANN::levenburg_marquardt_qr(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,jlen,i,jlenv,j,jlen2, i_off;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1;
	double lambda = 1000;
	double vraise = 10;
	double vreduce = 0.2;
	char line[MAXLINE];
	jlen = nsimr;
	jlenv = nsimv;
	if (doforces){
		jlen += natoms*3;
		jlenv += natoms*3;
	}
	jlen2 = jlen;
	if (doregularizer)jlen += betalen-1;//do not regulate last bias
	jlen1 = jlen+betalen;
//	double **J,**J1;
//	J = new double*[jlen+betalen];
//	J1 = new double*[jlen+betalen];
//	for (i=0;i<jlen+betalen;i++){
//		J[i] = new double[betalen];
//		J1[i] = new double[betalen];
//	}
//	double *target,*targetv,*target1;
//	target = new double [jlen+betalen];
//	target1 = new double [jlen+betalen];
//	targetv = new double [jlenv];
//	double *beta = new double[betalen];
//	double *D = new double[betalen];
//	double *delta = new double[betalen];
//	double *beta1 = new double[betalen];
	double J[jlen1*betalen];
	double J1[jlen1*betalen];
	double target[jlen1];
	double target1[jlen1];
	double targetv[jlenv];
	double beta[betalen];
	double beta1[betalen];
	double D[betalen];
	double *dp;
	double delta[jlen1];//extra length used internally in qrsolve
	dp = delta;
	double *Jp = J;
	double *Jp1 = J1;
    double *tp,*tp1,*bp,*bp1;
	tp = target;
	tp1 = target1;
	bp = beta;
	bp1 = beta1;
	force_fit = energy_fit = reg_fit = energy_fit1 = force_fit1 = reg_fit1 = 0.0;
//	sprintf(str,"%d\n",betalen);
//	std::cout<<str;
	clock_t start = clock();
	compute_jacobian(Jp,tp,r,nsimr,natomsr,net);
	clock_t end = clock();
	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	sprintf(str,"%f seconds\n",time);
	std::cout<<str;
//	for (i=0;i<nsimr;i++){
//		sprintf(str,"%d %f %.10f\n",i,tp[i],sims[i].energy);
//		std::cout<<str;
//	}

//	for (i=0;i<jlen-betalen;i++){
//		for (j=0;j<betalen;j++){
//			sprintf(str,"i %d j %d J %f",i,j,Jp[i*betalen+j]);
//			std::cout<<str;
//		}
//	}
	NNarchitecture net1[nelementsp];
	copy_network(net,net1);
	for (i=0;i<nsimr;i++){
		energy_fit += tp[i]*tp[i];
//		sprintf(str,"%d %f\n",i,tp[i]);
//		std::cout<<str;
	}
	energy_fit/=nsimr;
	if (doforces){
		for (i=nsimr;i<nsimr+natoms*3;i++)force_fit +=tp[i]*tp[i];
		force_fit/=natomsr*3;
	}
	if (doregularizer){
		for (i=1;i<betalen;i++){
			i_off = i+jlen-betalen;
			reg_fit +=tp[i_off]*tp[i_off];
		}
		reg_fit /= betalen;
	}
	flatten_beta(net,bp);
//	for (i=0;i<betalen;i++){
//		sprintf(str,"%d %f\n",i,bp[i]);
//		std::cout<<str;
//	}
	force_fitv = energy_fitv = 0.0;
	int counter = 0;
	int count1 = 0;
	iter = 0;
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces){
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	}
	else{
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
	}
	write_potential_file(true,line);
	for (i=0;i<betalen;i++){
		for (j=0;j<betalen;j++){
			Jp[(jlen+i)*betalen+j] = 0.0;
			Jp1[(jlen+i)*betalen+j] = 0.0;
		}
		tp1[jlen+i]=0.0;
		tp[jlen+i]=0.0;
	}
	while (iter<max_epochs){
		if (goodstep){
			if (nsimv>0){
				//do validation forward pass
				forward_pass(targetv,v,nsimv,net);
				//compute_jacobian(J1,targetv,v,nsimv,natomsv,net1);
				energy_fitv=0.0;
				for (i=0;i<nsimv;i++){
					energy_fitv += targetv[i]*targetv[i];
				}
				energy_fitv /= nsimv;
				if (doforces){
					force_fitv = 0.0;
					for (i=nsimv;i<natomsv*3+nsimv;i++){
						force_fitv += targetv[i]*targetv[i];
					}
					force_fitv/=(natomsv*3);
				}
			}
			else{
				energy_fitv = 0.0;
				force_fitv = 0.0;
			}
			for (i=0;i<betalen;i++){
				D[i] = 0.0;
				for (j=0;j<jlen2;j++){
					i_off = j*betalen+i;
					D[i] += J[i_off]*J[i_off];
				}
			}
		}
		if (doforces){
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		}
		else{
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
		}
		std::cout<<line;
		fprintf(fid,"%s",line);
		count1++;
		counter++;
		if (count1 == potential_output_freq){
			write_potential_file(true,line);
			count1 = 0;
		}
		for (i=0;i<betalen;i++){
			Jp[(jlen+i)*betalen+i] = sqrt(sqrt(D[i]*lambda));
		}
//		clock_t start = clock();
		qrsolve(Jp,jlen1,betalen,tp,dp);
//		clock_t end = clock();
//		double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
//		sprintf(str,"%f seconds\n",time);
//		std::cout<<str;
//		for (i=0;i<betalen;i++){
//			sprintf(str,"i %d delta %f\n",i,delta[i]);
//			std::cout<<str;
//		}
		for (i=0;i<betalen;i++)bp1[i]=bp[i]+dp[i];
		unflatten_beta(net1,bp1);
		compute_jacobian(Jp1,tp1,r,nsimr,natomsr,net1);
		energy_fit1 = 0.0;
		for (i=0;i<nsimr;i++)energy_fit1 += tp1[i]*tp1[i];
		energy_fit1/=nsimr;
		if (doforces){
			force_fit1 = 0.0;
			for (i=nsimr;i<natomsr*3+nsimr;i++){
				force_fit1 += tp1[i]*tp1[i];
			}
			force_fit1/=natomsr*3;
		}
		if (doregularizer){
			reg_fit1 = 0.0;
			for (i=1;i<betalen;i++){
				i_off = i+jlen-betalen;
				reg_fit1 +=tp1[i_off]*tp1[i_off];
			}
			reg_fit1 /= betalen;
		}
		if (energy_fit1+force_fit1+reg_fit1<energy_fit+force_fit+reg_fit){
			goodstep = true;
			lambda = lambda*vreduce;
			energy_fit = energy_fit1;
			force_fit = force_fit1;
			reg_fit = reg_fit1;
			double *tempb;
			tempb = bp;
			bp = bp1;
			bp1= tempb;
//			double **tempJ = J;
//			J = J1;
//			J1 = tempJ;
			double *tempJ;
			tempJ = Jp;
			Jp = Jp1;
			Jp1 = tempJ;
			double *tempT = tp;
			tp = tp1;
			tp1 = tempT;
			unflatten_beta(net,bp);
			iter++;
		}
		else {
			goodstep=false;
			lambda = lambda*vraise;
			if (lambda > 10e50){
				write_potential_file(true,line);
				errorf("Terminating because convergence is not making progress. Best fit output to file\n");
			}
		}
		if (energy_fit+force_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<=nelements;i++){
		if (net1[i].layers>0){
			for (int j=0;j<net1[i].layers-1;j++){
				delete [] net1[i].Weights[j];
				delete [] net1[i].Biases[j];
			}
			delete [] net1[i].dimensions;
			delete [] net1[i].dimensionsr;
			delete [] net1[i].Weights;
			delete [] net1[i].Biases;
			delete [] net1[i].startI;
		}
	}
}

//top level run function, calls compute_jacobian and qrsolve. Cannot be parallelized.
void PairRANN::levenburg_marquardt_ch(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,jlen,i,jlenv,j,jlen2;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1;
	double lambda = 1000;
	double vraise = 10;
	double vreduce = 0.2;
	char line[MAXLINE];

	int i_off, j_off, j_offPi;
	double time1, time2;

	jlen = nsimr;
	jlenv = nsimv;
	if (doforces){
		jlen += natoms*3;
		jlenv += natoms*3;
	}
	jlen2 = jlen;
	if (doregularizer)jlen += betalen;//do not regulate last bias
	jlen1 = jlen;
	double J[jlen1*betalen];
	double J1[jlen1*betalen];
	double J2[betalen*betalen];
	double t2[betalen];
	double target[jlen1];
	double target1[jlen1];
	double targetv[jlenv];
	double beta[betalen];
	double beta1[betalen];
	double D[betalen];
	double *dp;
	double delta[jlen1];//extra length used internally in qrsolve
	dp = delta;
//	double *Jp = J;
//	double *Jp1 = J1;
    double *tp,*tp1,*bp,*bp1,*Jp,*Jp1;
	tp = target;
	tp1 = target1;
	bp = beta;
	bp1 = beta1;
	Jp = J;
	Jp1 = J1;
	force_fit = energy_fit = reg_fit = energy_fit1 = force_fit1 = reg_fit1 = 0.0;
	sprintf(str,"types=%d; betalen=%d; jlen1=%d; jlen2=%d, regularization:%d\n",nelementsp,betalen,jlen1,jlen2, doregularizer);
	std::cout<<str;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	compute_jacobian(Jp,tp,r,nsimr,natomsr,net);


	NNarchitecture net1[nelementsp];
	copy_network(net,net1);
	for (i=0;i<nsimr;i++){
		energy_fit += tp[i]*tp[i];
	}
	energy_fit/=nsimr;
	if (doforces){
		for (i=nsimr;i<nsimr+natoms*3;i++)force_fit +=tp[i]*tp[i];
		force_fit/=natomsr*3;
	}
	if (doregularizer){
		for (i=1;i<betalen;i++){
			i_off = i+jlen-betalen;
			reg_fit +=tp[i_off]*tp[i_off];
		}
		reg_fit /= betalen;
	}
	flatten_beta(net,bp);
	force_fitv = energy_fitv = 0.0;
	int counter = 0;
	int count1 = 0;
	iter = 0;
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces){
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	}
	else{
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
	}
	write_potential_file(true,line);
	double start2;
	while (iter<max_epochs){
		if (goodstep){
			if (nsimv>0){
				//do validation forward pass
				forward_pass(targetv,v,nsimv,net);
				//compute_jacobian(J1,targetv,v,nsimv,natomsv,net1);
				energy_fitv=0.0;
				for (i=0;i<nsimv;i++){
					energy_fitv += targetv[i]*targetv[i];
				}
				energy_fitv /= nsimv;
				if (doforces){
					force_fitv = 0.0;
					for (i=nsimv;i<natomsv*3+nsimv;i++){
						force_fitv += targetv[i]*targetv[i];
					}
					force_fitv/=(natomsv*3);
				}
			}
			else{
				energy_fitv = 0.0;
				force_fitv = 0.0;
			}

			// clock_t start2 = clock();
			start2 = omp_get_wtime();

			for (i=0;i<betalen;i++){
				i_off = i*betalen;
				for (int k=0;k<=i;k++){
					J2[i_off+k] = 0.0;
				}
			}
			//major bottleneck here:
			//need to rewrite matrix multiplication to use cache better
			//https://en.wikipedia.org/wiki/Loop_nest_optimization

			// for (int j=0;j<jlen2;j++){
			// 	int j_off = j*betalen;
			// 	for (int i=0;i<betalen;i++){
			// 		int j_offPi = j_off+i;
			// 		int i_off = i*betalen;
			// 		for (int k=0;k<=i;k++){
			// 			J2[i_off+k] += Jp[j_offPi]*Jp[j_off+k];
			// 		}
			// 	}
			// }

			// #pragma omp parallel default(none) shared(J2,Jp,jlen2,betalen, doregularizer)
			#pragma omp parallel
			{
			// loop reordered to remove the dependancy. Single thread calculation would be slow. Observed gain when thread more than around 8
			#pragma omp for
			for (int i=0;i<betalen;i++){
				int i_off = i*betalen;
				for (int k=0;k<=i;k++){
					for (int j=0;j<jlen2;j++){
						int j_off = j*betalen;
						int j_offPi = j_off+i;
						J2[i_off+k] += Jp[j_offPi]*Jp[j_off+k];
					}
				}
			}

			if (doregularizer){
				#pragma omp for
				for (int i=0;i<betalen;i++){
					int	i_off = i*betalen;
					int ij_off = jlen2*betalen + i_off;
					J2[i_off+i]+=Jp[ij_off+i]*Jp[ij_off+i];
				}
			}

			#pragma omp barrier
			#pragma omp single
			{
			for (int i=0;i<betalen;i++){
				D[i] = J2[i*betalen+i];
//				printf("%d %f\n",i,D[i]);
				if (D[i]==0){errorf("Jacobian is rank deficient!\n");}
				if (doregularizer) // t2 can be initialized with 0 or derivative w.r.t. weight
					t2[i]=Jp[jlen2*betalen+i*betalen+i]*tp[jlen2+i];
				else
				    t2[i]=0;
			}
			}

			// loop splitting for threading. Initialization for t2 is done above.
			#pragma omp for
			for (int i=0;i<betalen;i++){
				// t2[i]=0;
				for (j=0;j<jlen2;j++){
					t2[i]+=Jp[j*betalen+i]*tp[j];
				}
			}

			}
			// if (doregularizer){
			// 	for (int i=0;i<betalen;i++){
			// 		t2[i]+=Jp[jlen2*betalen+i*betalen+i]*tp[jlen2+i];
			// 	}
			// }

			// clock_t end = clock();
			// time2 = (double) (end-start2) / CLOCKS_PER_SEC * 1000.0;
			// sprintf(str,"loop: %f ms\n",time2);
			// std::cout<<str;
			double time = (double) (omp_get_wtime() - start2)*1000.0;
			printf("loop: %f ms\n",time);

		}
		if (doforces){
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		}
		else{
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
		}
		std::cout<<line;
		fprintf(fid,"%s",line);
		count1++;
		counter++;
		if (count1 == potential_output_freq){
			write_potential_file(true,line);
			count1 = 0;
		}
		for (i=0;i<betalen;i++){
			J2[i*betalen+i]=D[i]+sqrt(D[i]*lambda);
		}
//		clock_t start1 = clock();
		chsolve(J2,betalen,t2,dp);
//		clock_t end1 = clock();
//		time = (double) (end1-start1) / CLOCKS_PER_SEC * 1000.0;
//		sprintf(str,"chsolve(): %f ms\n",time);
//		std::cout<<str;

//		for (i=0;i<betalen;i++){
//			sprintf(str,"i %d delta %f\n",i,delta[i]);
//			std::cout<<str;
//		}
		for (i=0;i<betalen;i++)bp1[i]=bp[i]+dp[i];
		unflatten_beta(net1,bp1);
		compute_jacobian(Jp1,tp1,r,nsimr,natomsr,net1);
		energy_fit1 = 0.0;
		for (i=0;i<nsimr;i++)energy_fit1 += tp1[i]*tp1[i];
		energy_fit1/=nsimr;
		if (doforces){
 			force_fit1 = 0.0;
			for (i=nsimr;i<natomsr*3+nsimr;i++){
				force_fit1 += tp1[i]*tp1[i];
			}
			force_fit1/=natomsr*3;
		}
		if (doregularizer){
			reg_fit1 = 0.0;
			for (i=1;i<betalen;i++){
				i_off = i+jlen-betalen;
				reg_fit1 += tp1[i_off]*tp1[i_off];
			}
			reg_fit1 /= betalen;
		}
		if (energy_fit1+force_fit1+reg_fit1<energy_fit+force_fit+reg_fit){
			goodstep = true;
			lambda = lambda*vreduce;
			energy_fit = energy_fit1;
			force_fit = force_fit1;
			reg_fit = reg_fit1;
			double *tempb;
			tempb = bp;
			bp = bp1;
			bp1= tempb;
			double *tempJ;
			tempJ = Jp;
			Jp = Jp1;
			Jp1 = tempJ;
			double *tempT = tp;
			tp = tp1;
			tp1 = tempT;
			unflatten_beta(net,bp);
			iter++;
		}
		else {
			goodstep=false;
			lambda = lambda*vraise;
			if (lambda > 10e50){
				write_potential_file(true,line);
				errorf("Terminating because convergence is not making progress. Best fit output to file\n");
			}
		}
		if (energy_fit+force_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			write_potential_file(true,line);
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<=nelements;i++){
		if (net1[i].layers>0){
			for (int j=0;j<net1[i].layers-1;j++){
				delete [] net1[i].Weights[j];
				delete [] net1[i].Biases[j];
			}
			delete [] net1[i].dimensions;
			delete [] net1[i].dimensionsr;
			delete [] net1[i].Weights;
			delete [] net1[i].Biases;
			delete [] net1[i].startI;
		}
	}

	// clock_t end = clock();
    // time1 = (double) (end-start1) / CLOCKS_PER_SEC * 1000.0;
	// sprintf(str,"LM_ch(): %f ms\n",time1);
	// std::cout<<str;
    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
    printf("LM_ch(): %f ms\n",time);

}

void PairRANN::flatten_beta(NNarchitecture *net,double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
	for (itype=0;itype<nelementsp;itype++){
		for (i=1;i<net[itype].layers;i++){
			for (k1=0;k1<net[itype].dimensions[i];k1++){
				for (k2=net[itype].dimensionsr[i-1];k2<net[itype].dimensions[i-1];k2++){
					beta[count2]=net[itype].Weights[i-1][k1*net[itype].dimensions[i-1]+k2];
					count2++;
				}
				if (net[itype].dimensionsr[i-1]==0){
					beta[count2]=net[itype].Biases[i-1][k1];
					count2++;
				}
			}
		}
	}
}

void PairRANN::unflatten_beta(NNarchitecture *net,double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
	char str[MAXLINE];
	for (itype=0;itype<nelementsp;itype++){
		for (i=1;i<net[itype].layers;i++){
			for (k1=0;k1<net[itype].dimensions[i];k1++){
				for (k2=net[itype].dimensionsr[i-1];k2<net[itype].dimensions[i-1];k2++){
					net[itype].Weights[i-1][k1*net[itype].dimensions[i-1]+k2]=beta[count2];
					count2++;
				}
				if (net[itype].dimensionsr[i-1]==0){
					net[itype].Biases[i-1][k1]=beta[count2];
					count2++;
				}
			}
		}
	}
}

//this is the main candidate for parallel distribution. Loop order does not matter. Outermost loop is probably best distribution point:
//Each loop writes one row of the matrix (multiple rows if doforces is on)
//row order also does not matter as long as it matches the corresponding value in target
void PairRANN::compute_jacobian(double *J,double *target,int *s,int sn,int natoms,NNarchitecture *net){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	#pragma omp parallel
	{
//	char str[MAXLINE];
	int nn,ii,n1;
	int count4 = 0;
	int n1dimi, n1sl, n1slM1, n4s, lM1, pIPk2, pLPk2;
	int sPcPiiX3, p1dlxyz, p2dlxyz, jPstartI, jjXfPk, iiX3, j1X3, p1dXw, p2dXw, p1ddXw, p2ddXw, i2n1W;

	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = sims[nn].inum;
		double energy;
		double force[n4s*3];
		energy = 0.0;
		for (ii=0;ii<betalen;ii++){
			J[n1*betalen+ii]=0.0;
		}
		if (doforces){
			for (ii=0;ii<3*n4s;ii++){
				force[ii]=0.0;
			}
		}
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,startI,prevI,l,startL,prevL,k1,k2,k3;
			startI=0;
			NNarchitecture net1;
			itype = sims[nn].type[ii];
			net1 = net[itype];
			n1sl = net1.sumlayers;
			n1slM1 = n1sl-1;
			iiX3 = ii*3;
			sPcPiiX3 = sn+count4+iiX3;
			numneigh = sims[nn].numneigh[ii];
			jnum = numneigh+1;//extra value on the end of the array is the self term.
			firstneigh = sims[nn].firstneigh;
			jlist = firstneigh[ii];
			int L = net1.layers-1;
			double layer[n1sl];
			double dlayer[n1sl];
			double ddlayer[n1sl];
			double dlayerx[jnum*n1sl];
			double dlayery[jnum*n1sl];
			double dlayerz[jnum*n1sl];
			int f = net1.dimensions[0];
			double *features = sims[nn].features[ii];
			double *dfeaturesx;
			double *dfeaturesy;
			double *dfeaturesz;
			if (doforces){
				dfeaturesx = sims[nn].dfx[ii];
				dfeaturesy = sims[nn].dfy[ii];
				dfeaturesz = sims[nn].dfz[ii];
			}
			prevI = 0;
			for (i=0;i<net1.layers-1;i++){
				n1dimi = net1.dimensions[i];
				for (j=0;j<net1.dimensions[i+1];j++){
					//energy forward propagation
					startI = net1.startI[i+1];
					jPstartI = j+startI;
					layer[jPstartI]=0;
					for (k=0;k<n1dimi;k++){
						if (i==0&&j==0){
							layer[k]=features[k];
							dlayer[k] = 1.0;
							ddlayer[k] = 0.0;
						}
						layer[jPstartI] += net1.Weights[i][j*n1dimi+k]*layer[k+prevI];
					}
					layer[jPstartI] += net1.Biases[i][j];
					if (doforces){
						ddlayer[jPstartI] = activation[itype][i]->ddactivation_function(layer[jPstartI]);
					}
					dlayer[jPstartI] = activation[itype][i]->dactivation_function(layer[jPstartI]);
					layer[jPstartI] =  activation[itype][i]-> activation_function(layer[jPstartI]);
//					if (n1==1){
//						sprintf(str,"%f\n",layer[jPstartI]);
//						std::cout<<str;
//					}
					if (i==L-1){
						energy += layer[jPstartI];
					}
					if (doforces){
						//force forward propagation
						for (jj=0;jj<jnum;jj++){
							p1dlxyz = jj*n1sl+jPstartI;
							dlayerx[p1dlxyz]=0;
							dlayery[p1dlxyz]=0;
							dlayerz[p1dlxyz]=0;
							for (k=0;k<n1dimi;k++){
								if (i==0&&j==0){
									p2dlxyz = jj*n1sl+k;
									jjXfPk = jj*f+k;
									dlayerx[p2dlxyz]=dfeaturesx[jjXfPk];
									dlayery[p2dlxyz]=dfeaturesy[jjXfPk];
									dlayerz[p2dlxyz]=dfeaturesz[jjXfPk];
								}
								double w1 = net1.Weights[i][j*n1dimi+k];
								p2dlxyz = jj*n1sl+k+prevI;
								dlayerx[p1dlxyz] += w1*dlayerx[p2dlxyz];
								dlayery[p1dlxyz] += w1*dlayery[p2dlxyz];
								dlayerz[p1dlxyz] += w1*dlayerz[p2dlxyz];
							}
							dlayerx[p1dlxyz] *= dlayer[jPstartI];
							dlayery[p1dlxyz] *= dlayer[jPstartI];
							dlayerz[p1dlxyz] *= dlayer[jPstartI];
							if (i==L-1 && jj < (jnum-1)){
								j1 = jlist[jj];
								j1 &= NEIGHMASK;
								j1X3 = j1*3;
								force[j1X3  ]+=dlayerx[p1dlxyz];
								force[j1X3+1]+=dlayery[p1dlxyz];
								force[j1X3+2]+=dlayerz[p1dlxyz];
							}
						}
						if (i==L-1){
							j1 = ii;
							jj = jnum-1;
							j1X3 = j1*3;
							p1dlxyz = jj*n1sl+jPstartI;
							force[j1X3  ]+=dlayerx[p1dlxyz];
							force[j1X3+1]+=dlayery[p1dlxyz];
							force[j1X3+2]+=dlayerz[p1dlxyz];
						}
					}
				}
				prevI=startI;
			}
			prevI=0;
			int count2=0;
			if (itype>0){
				count2=betalen_v[itype-1];
			}
			int count3=0;
			//backpropagation
			for (i=1;i<net1.layers;i++){
				n1dimi = net1.dimensions[i];
				double dXw[n1sl*n1dimi];
				startI = net1.startI[i];
				for (k1=0;k1<n1sl;k1++){
					for (k2=0;k2<n1dimi;k2++){
						p1dXw = k1*n1dimi+k2;
						dXw[p1dXw]=0.0;
						if (k1==k2+startI){
							dXw[p1dXw]=1.0;
						}
					}
				}
				for (l=i+1;l<net1.layers;l++){
					prevL = net1.startI[l-1];
					startL = net1.startI[l];
					for (k1=0;k1<net1.dimensions[l];k1++){
						for (k2=0;k2<net1.dimensions[l-1];k2++){
							pLPk2 = prevL+k2;
							for (k3=0;k3<n1dimi;k3++){
								p1dXw = (k1+startL)*n1dimi+k3;
								p2dXw = (pLPk2)*n1dimi+k3;
								dXw[p1dXw]+=net1.Weights[l-1][k1*net1.dimensions[l-1]+k2]*dlayer[pLPk2]*dXw[p2dXw];
							}
						}
					}
				}

				//fill nn row of Jacobian
				for (k1=0;k1<n1dimi;k1++){
					p2dXw = n1slM1*n1dimi+k1;
					for (k2=net1.dimensionsr[i-1];k2<net1.dimensions[i-1];k2++){
						J[n1*betalen+count2] += -dXw[p2dXw]*layer[k2+prevI]*sims[nn].energy_weight;
						count2++;
					}
					if (net[itype].dimensionsr[i-1]==0){
						J[n1*betalen+count2] += -dXw[p2dXw]*sims[nn].energy_weight;
						count2++;
					}
				}
				//force backpropagation
				if (doforces){	//do dimsreserved
					k3 = n1sl*n1dimi*jnum; // Z[L,N,M]
					double ddXwx[k3];
					double ddXwy[k3];
					double ddXwz[k3];
					for (k1=0;k1<k3;k1++){
						ddXwx[k1]=0.0;
						ddXwy[k1]=0.0;
						ddXwz[k1]=0.0;
					}
					for (l=i+1;i<net1.layers;l++){
						lM1 = l-1;
						prevL = net1.startI[lM1];
						startL = net1.startI[l];
						for (k1=0;k1<net1.dimensions[l];k1++){
							for (k2=0;k2<net1.dimensions[lM1];k2++){
								i2n1W = k1*net1.dimensions[lM1]+k2;
								pLPk2 = prevL+k2;
								for (k3=0;k3<n1dimi;k3++){
									p1dXw = (k1+startL)*n1dimi+k3;
									p2dXw = pLPk2*n1dimi+k3;
									for (jj=0;jj<jnum;jj++){
										p1ddXw = p1dXw*jnum+jj;
										p2ddXw = p2dXw*jnum+jj;
										p1dlxyz = jj*n1sl+pLPk2;
										ddXwx[p1ddXw] += net1.Weights[lM1][i2n1W]*ddlayer[pLPk2]*dXw[p2dXw]*dlayerx[p1dlxyz]+net1.Weights[lM1][i2n1W]*dlayer[pLPk2]*ddXwx[p2ddXw];
										ddXwy[p1ddXw] += net1.Weights[lM1][i2n1W]*ddlayer[pLPk2]*dXw[p2dXw]*dlayery[p1dlxyz]+net1.Weights[lM1][i2n1W]*dlayer[pLPk2]*ddXwy[p2ddXw];
										ddXwz[p1ddXw] += net1.Weights[lM1][i2n1W]*ddlayer[pLPk2]*dXw[p2dXw]*dlayerz[p1dlxyz]+net1.Weights[lM1][i2n1W]*dlayer[pLPk2]*ddXwz[p2ddXw];
									}
								}
							}
						}
					}
					for (k1=net1.dimensionsr[i];k1<n1dimi;k1++){
						p1dXw = n1slM1*n1dimi+k1;
						for (k2=net1.dimensionsr[i-1];k2<net1.dimensions[i-1];k2++){
							pIPk2 = prevI+k2;
							for (jj=0;jj<jnum;jj++){
								p1ddXw = p1dXw*jnum+jlist[jj];
								p1dlxyz = jj*n1sl+pIPk2;
								J[(sPcPiiX3  )*betalen+count3] = (ddXwx[p1ddXw]*layer[pIPk2]+dXw[p1dXw]*dlayer[pIPk2]*dlayerx[p1dlxyz])*sims[nn].force_weight;
								J[(sPcPiiX3+1)*betalen+count3] = (ddXwy[p1ddXw]*layer[pIPk2]+dXw[p1dXw]*dlayer[pIPk2]*dlayery[p1dlxyz])*sims[nn].force_weight;
								J[(sPcPiiX3+2)*betalen+count3] = (ddXwz[p1ddXw]*layer[pIPk2]+dXw[p1dXw]*dlayer[pIPk2]*dlayerz[p1dlxyz])*sims[nn].force_weight;
							}
							count3++;
						}
						if (net[itype].dimensionsr[i-1]==0){
							for (jj=0;jj<jnum;jj++){
								p1ddXw = p1dXw*jnum+jlist[jj];
								J[(sPcPiiX3  )*betalen+count3] = ddXwx[p1ddXw]*sims[nn].force_weight;
								J[(sPcPiiX3+1)*betalen+count3] = ddXwy[p1ddXw]*sims[nn].force_weight;
								J[(sPcPiiX3+2)*betalen+count3] = ddXwz[p1ddXw]*sims[nn].force_weight;
							}
							count3++;
						}
					}
				}
				prevI=startI;
			}
		}
		//fill error vector
		target[n1] = (energy-sims[nn].energy)*sims[nn].energy_weight;
		if (doforces){
			for (ii=0;ii<n4s;ii++){
				iiX3 = ii*3;
				sPcPiiX3 = sn+count4+iiX3;
				target[sPcPiiX3  ] = (force[iiX3  ]-sims[nn].f[ii][0])*sims[nn].force_weight;
				target[sPcPiiX3+1] = (force[iiX3+1]-sims[nn].f[ii][1])*sims[nn].force_weight;
				target[sPcPiiX3+2] = (force[iiX3+2]-sims[nn].f[ii][2])*sims[nn].force_weight;
			}
		}
		count4 += n4s*3;
	}
	//regularizer
	if (doregularizer){
		int count2 = 0;
		if (doforces){
			// #pragma omp for schedule(dynamic)
			for (int itype=0;itype<nelementsp;itype++){
				for (int i=1;i<net[itype].layers;i++){
					for (int k1=0;k1<net[itype].dimensions[i];k1++){
						for (int k2=net[itype].dimensionsr[i-1];k2<net[itype].dimensions[i-1];k2++){
							J[(sn+natoms*3+count2)*betalen+count2] = regularizer;
							target[sn+natoms*3+count2] = -regularizer*net[itype].Weights[i-1][k1*net[itype].dimensions[i-1]+k2];
							count2++;
						}
						if (net[itype].dimensionsr[i-1]==0){
							J[(sn+natoms*3+count2)*betalen+count2] = regularizer;
							target[sn+natoms*3+count2] = -regularizer*net[itype].Biases[i-1][k1];
							if (i+1==net[itype].layers){
								J[(sn+natoms*3+count2)*betalen+count2]=0;
								target[sn+natoms*3+count2]=0;
							}//do not regulate last bias
							count2++;
						}
					}
				}
			}
		}
		else{
			// #pragma omp for schedule(dynamic)
			for (int itype=0;itype<nelementsp;itype++){
				for (int i=1;i<net[itype].layers;i++){
					for (int k1=0;k1<net[itype].dimensions[i];k1++){
						for (int k2=net[itype].dimensionsr[i-1];k2<net[itype].dimensions[i-1];k2++){
							J[(sn+count2)*betalen+count2] = regularizer;
//							printf("%d %f",(sn+count2)*betalen+count2,J[(sn+count2)*betalen+count2]);
							target[sn+count2] = -regularizer*net[itype].Weights[i-1][k1*net[itype].dimensions[i-1]+k2];
							count2++;
						}
						if (net[itype].dimensionsr[i-1]==0){
							J[(sn+count2)*betalen+count2] = regularizer;
							target[sn+count2] = -regularizer*net[itype].Biases[i-1][k1];
							if (i+1==net[itype].layers){
								J[(sn+count2)*betalen+count2]=0;
								target[sn+count2]=0;
							}//do not regulate last bias
//							printf("%d %f",(sn+count2)*betalen+count2,J[(sn+count2)*betalen+count2]);
							count2++;
						}
					}
				}
			}
		}
	}

    }

//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	printf(" - compute_jacobian(): %f ms\n",time);

}

void PairRANN::forward_pass(double *target,int *s,int sn,NNarchitecture *net){
	//clock_t start = clock();
	double start_time = omp_get_wtime();
	#pragma omp parallel
	{
	int nn,ii,n1;
	int jPstartI, jjXfPk, n1sl, n1slM1, p1dlxyz, p2dlxyz, sPcPiiX3, n4s, iiX3, j1X3;
	int count4 = 0;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = sims[nn].inum;
		double energy;
		double force[n4s*3];
		energy = 0.0;
		if (doforces){
			for (ii=0;ii<3*n4s;ii++){
				force[ii]=0.0;
			}
		}
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,startI,prevI;
			startI=0;
			NNarchitecture net1;
			itype = sims[nn].type[ii];
			net1 = net[itype];
			n1sl = net1.sumlayers;
			n1slM1 = n1sl-1;
			numneigh = sims[nn].numneigh[ii];
			jnum = numneigh+1;//extra value on the end of the array is the self term.
			firstneigh = sims[nn].firstneigh;
			jlist = firstneigh[ii];
			int L = net1.layers-1;
			double layer[n1sl];
			double dlayer[n1sl];
			double dlayerx[jnum*n1sl];
			double dlayery[jnum*n1sl];
			double dlayerz[jnum*n1sl];
			int f = net1.dimensions[0];
			double *features = sims[nn].features[ii];
			double *dfeaturesx;
			double *dfeaturesy;
			double *dfeaturesz;
			if (doforces){
				dfeaturesx = sims[nn].dfx[ii];
				dfeaturesy = sims[nn].dfy[ii];
				dfeaturesz = sims[nn].dfz[ii];
			}
			prevI = 0;
			for (i=0;i<net1.layers-1;i++){
				for (j=0;j<net1.dimensions[i+1];j++){
					//energy forward propagation
					startI = net1.startI[i+1];
					jPstartI = j+startI;
					layer[jPstartI]=0;
					for (k=0;k<net1.dimensions[i];k++){
						if (i==0&&j==0){
							layer[k]=features[k];
							dlayer[k] = 1.0;
						}
						layer[jPstartI] += net1.Weights[i][j*net1.dimensions[i]+k]*layer[k+prevI];
					}
					layer[jPstartI] += net1.Biases[i][j];
					dlayer[jPstartI] = activation[itype][i]->dactivation_function(layer[jPstartI]);
					layer[jPstartI] =  activation[itype][i]-> activation_function(layer[jPstartI]);
					if (i==L-1){
						energy += layer[jPstartI];
					}
					if (doforces){
						//force forward propagation
						for (jj=0;jj<jnum;jj++){
							p1dlxyz = jj*n1sl+jPstartI;
							dlayerx[p1dlxyz]=0;
							dlayery[p1dlxyz]=0;
							dlayerz[p1dlxyz]=0;
							for (k=0;k<net1.dimensions[i];k++){
								if (i==0&&j==0){
									jjXfPk = jj*f+k;
									p2dlxyz = jj*n1sl+k;
									dlayerx[p2dlxyz]=dfeaturesx[jjXfPk];
									dlayery[p2dlxyz]=dfeaturesy[jjXfPk];
									dlayerz[p2dlxyz]=dfeaturesz[jjXfPk];
								}
								double w1 = net1.Weights[i][j*net1.dimensions[i]+k];
								p2dlxyz = jj*n1sl+k+prevI;
								dlayerx[p1dlxyz] += w1*dlayerx[p2dlxyz];
								dlayery[p1dlxyz] += w1*dlayery[p2dlxyz];
								dlayerz[p1dlxyz] += w1*dlayerz[p2dlxyz];
							}
							dlayerx[p1dlxyz] *= dlayer[jPstartI];
							dlayery[p1dlxyz] *= dlayer[jPstartI];
							dlayerz[p1dlxyz] *= dlayer[jPstartI];
							if (i==L-1 && jj < (jnum-1)){
								j1 = jlist[jj];
								j1 &= NEIGHMASK;
								j1X3 = j1*3;
								force[j1X3  ]+=dlayerx[p1dlxyz];
								force[j1X3+1]+=dlayery[p1dlxyz];
								force[j1X3+2]+=dlayerz[p1dlxyz];
							}
						}
						if (i==L-1){
							j1 = ii;
							j1X3 = j1*3;
							jj = jnum-1;
							p1dlxyz = jj*n1sl+jPstartI;
							force[j1X3  ]+=dlayerx[p1dlxyz];
							force[j1X3+1]+=dlayery[p1dlxyz];
							force[j1X3+2]+=dlayerz[p1dlxyz];
						}
					}
				}
				prevI=startI;
			}
			prevI=0;
			//fill error vector
			target[n1] = (energy-sims[nn].energy)*sims[nn].energy_weight;
			if (doforces){
				for (ii=0;ii<n4s;ii++){
					iiX3 = ii*3;
					sPcPiiX3 = sn+count4+iiX3;
					target[sPcPiiX3  ] = (force[iiX3  ]-sims[nn].f[ii][0])*sims[nn].force_weight;
					target[sPcPiiX3+1] = (force[iiX3+1]-sims[nn].f[ii][1])*sims[nn].force_weight;
					target[sPcPiiX3+2] = (force[iiX3+2]-sims[nn].f[ii][2])*sims[nn].force_weight;
				}
			}
		}
		count4 += n4s*3;
	}
	}
//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	printf(" - forward_pass(): %f ms\n",time);

}

void PairRANN::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn){
	int *jlist,j,count,jj,*type,jtype;
	double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double **x = sims[sn].x;
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	type = sims[sn].type;
	jlist = sims[sn].firstneigh[i];
	count = 0;
	for (jj=0;jj<jnum[0];jj++){
		j = jlist[jj];
		j &= NEIGHMASK;
		jtype = map[type[j]];
		delx = xtmp - x[j][0];
		dely = ytmp - x[j][1];
		delz = ztmp - x[j][2];
		rsq = delx*delx + dely*dely + delz*delz;
		if (rsq>cutmax*cutmax){
			continue;
		}
		xn[count]=delx;
		yn[count]=dely;
		zn[count]=delz;
		tn[count]=jtype;
		jl[count]=sims[sn].id[j];//j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
		count++;
	}
	jnum[0]=count+1;
}

void PairRANN::screen_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,bool *Bij,double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz){
	double xnc[jnum[0]],ync[jnum[0]],znc[jnum[0]];
	double Sikc[jnum[0]];
	double dSikxc[jnum[0]];
	double dSikyc[jnum[0]];
	double dSikzc[jnum[0]];
	double dSijkxc[jnum[0]][jnum[0]];
	double dSijkyc[jnum[0]][jnum[0]];
	double dSijkzc[jnum[0]][jnum[0]];
	int jj,kk,count,count1,tnc[jnum[0]],jlc[jnum[0]];
	count = 0;
	for (jj=0;jj<jnum[0]-1;jj++){
		if (Bij[jj]){
			count1 = 0;
			xnc[count]=xn[jj];
			ync[count]=yn[jj];
			znc[count]=zn[jj];
			tnc[count]=tn[jj];
			jlc[count]=jl[jj];
			Sikc[count]=Sik[jj];
			dSikxc[count]=dSikx[jj];
			dSikyc[count]=dSiky[jj];
			dSikzc[count]=dSikz[jj];
			for (kk=0;kk<jnum[0]-1;kk++){
				if (Bij[kk]){
					dSijkxc[count][count1] = dSijkx[jj*(jnum[0]-1)+kk];
					dSijkyc[count][count1] = dSijky[jj*(jnum[0]-1)+kk];
					dSijkzc[count][count1] = dSijkz[jj*(jnum[0]-1)+kk];
					count1++;
				}
			}
			count++;
		}
	}
	jnum[0]=count+1;
	for (jj=0;jj<count;jj++){
		xn[jj]=xnc[jj];
		yn[jj]=ync[jj];
		zn[jj]=znc[jj];
		tn[jj]=tnc[jj];
		jl[jj]=jlc[jj];
		Bij[jj] = true;
		Sik[jj]=Sikc[jj];
		dSikx[jj]=dSikxc[jj];
		dSiky[jj]=dSikyc[jj];
		dSikz[jj]=dSikzc[jj];
		for (kk=0;kk<count;kk++){
			dSijkx[jj*count+kk] = dSijkxc[jj][kk];
			dSijky[jj*count+kk] = dSijkyc[jj][kk];
			dSijkz[jj*count+kk] = dSijkzc[jj][kk];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//replaced with Cholesky solution for greater speed.
void PairRANN::qrsolve(double *A,int m,int n,double *b, double *x_){
	double QR_[m*n];
//	char str[MAXLINE];
	double Rdiag[n];
	int i=0, j=0, k=0;
	int j_off, k_off;
	double nrm;
    // loop to copy QR from A.
	for (k=0;k<n;k++){
		k_off = k*m;
		for (i=0;i<m;i++){
			QR_[k_off+i]=A[i*n+k];
		}
	}
    for (k = 0; k < n; k++) {
       // Compute 2-norm of k-th column.
       nrm = 0.0;
       k_off = k*m;
       for (i = k; i < m; i++) {
			nrm += QR_[k_off+i]*QR_[k_off+i];
       }
       if (nrm==0.0){
    	   errorf("Jacobian is rank deficient!\n");
       }
       nrm = sqrt(nrm);
	   // Form k-th Householder vector.
	   if (QR_[k_off+k] < 0) {
		 nrm = -nrm;
 	   }
	   for (i = k; i < m; i++) {
		 QR_[k_off+i] /= nrm;
	   }
	   QR_[k_off+k] += 1.0;

	   // Apply transformation to remaining columns.
	   for (j = k+1; j < n; j++) {
		 double s = 0.0;
		 j_off = j*m;
		 for (i = k; i < m; i++) {
			s += QR_[k_off+i]*QR_[j_off+i];
		 }
		 s = -s/QR_[k_off+k];
		 for (i = k; i < m; i++) {
			QR_[j_off+i] += s*QR_[k_off+i];
		 }
	   }
       Rdiag[k] = -nrm;
    }
    //loop to find least squares
    for (int j=0;j<m;j++){
    	x_[j] = b[j];
    }
    // Compute Y = transpose(Q)*b
	for (int k = 0; k < n; k++)
	{
		k_off = k*m;
		double s = 0.0;
		for (int i = k; i < m; i++)
		{
		   s += QR_[k_off+i]*x_[i];
		}
		s = -s/QR_[k_off+k];
		for (int i = k; i < m; i++)
		{
		   x_[i] += s*QR_[k_off+i];
		}
	}
	// Solve R*X = Y;
	for (int k = n-1; k >= 0; k--)
	{
		k_off = k*m;
		x_[k] /= Rdiag[k];
		for (int i = 0; i < k; i++) {
		   x_[i] -= x_[k]*QR_[k_off+i];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//should be optimized for better cache use, possibly add openmp
/*
void PairRANN::chsolve(double *A,int n,double *b, double *x){
	double L_[n*n]; // was L_[n][n]
	int i,j,k;
	int jXn, kXn;
	double d, s;

	clock_t start1 = clock();

//	for (j=0;j<n;j++){
//		for (k=0;k<n;k++){
//			L_[j][k]=0.0;
//		}
//	}
	for (k=0;k<n*n;k++){
		L_[k]=0.0;
	}
	// Main loop.
	for (j = 0; j < n; j++)
	{
		d=0.0;;
		jXn = j*n;
		for (k = 0; k < j; k++)
		{
			s=0;
			kXn = k*n;
			for (i = 0; i < k; i++)
			{
//				s += L_[k][i]*L_[j][i];
				s += L_[kXn+i]*L_[jXn+i];
			}
//			L_[j][k] = s = (A[j*n+k] - s)/L_[k][k];
			L_[jXn+k] = s = (A[jXn+k] - s)/L_[kXn+k];
			d = d + s*s;
		}
		d = A[jXn+j] - d;
//		L_[j][j] = 0.0;
		L_[jXn+j] = 0.0;
		if (d>0){
//			L_[j][j] = sqrt(d);
			L_[jXn+j] = sqrt(d);
		}
//		L_[j][j] = sqrt(d > 0.0 ? d : 0.0);
		for (k = j+1; k < n; k++)
		{
//			L_[j][k] = 0.0;
			L_[jXn+k] = 0.0;
		}
	}
    for (j=0;j<n;j++){
    	x[j] = b[j];
    }
	// Solve L*y = b;
	for (k = 0; k < n; k++)
	{
//		for (i = 0; i < k; i++) x[k] -= x[i]*L_[k][i];
//		x[k] /= L_[k][k];
		kXn = k*n;
		for (i = 0; i < k; i++) x[k] -= x[i]*L_[kXn+i];
		x[k] /= L_[kXn+k];
	}
	// Solve L'*X = Y;
	for (k = n-1; k >= 0; k--)
	{
//		for (i = k+1; i < n; i++) x[k] -= x[i]*L_[i][k];
//		x[k] /= L_[k][k];
		for (i = k+1; i < n; i++) x[k] -= x[i]*L_[i*n+k];
		x[k] /= L_[k*n+k];
	}

	clock_t end1 = clock();
	double time = (double) (end1-start1) / CLOCKS_PER_SEC * 1000.0;
	printf(" - chsolve(): %f ms\n",time);

	return;
}
*/
void PairRANN::chsolve(double *A,int n,double *b, double *x){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	int	nthreads=omp_get_num_threads();

	double L_[n*n]; // was L_[n][n]
	int i,j,k;
	int iXn, jXn, kXn;
	double d, s;

	// initialize L
	for (k=0;k<n*n;k++){
		L_[k]=0.0;
	}

	// Cholesky-Crout decomposition
	#pragma omp parallel default(none) shared (A,L_,n,s)
	{
	for (int j = 0; j <n; j++) {
		int jXn = j*n;
		s = 0.0;
		// #pragma omp for schedule(static) reduction(+:s)
		for (int k = 0; k < j; k++) {
			s += L_[jXn + k] * L_[jXn + k];
		}
		#pragma omp barrier
		double d = A[jXn+j] - s;
		#pragma omp single
		{
		if (d>0){
			L_[jXn + j] = sqrt(d);
		}
		}
		//// #pragma omp parallel for schedule(static) default(none) shared (A,L_,n,j,jXn)
		////#pragma omp barrier
		#pragma omp for schedule(static)
		for (int i = j+1; i <n; i++) {
			int iXn = i * n;
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += L_[iXn + k] * L_[jXn + k];
			}
			L_[iXn + j] =  (A[iXn + j] - sum) / L_[jXn + j];
		}
	}
	}
	// Solve L*
	// Forward substitution to solve L*y = b;
	// #pragma omp parallel default(none) shared (x,b,L_,n,s) private(i)
	// #pragma omp parallel
	{
	for (int k = 0; k < n; k++)
	{
		int kXn = k*n;
		s = 0.0;
		// #pragma omp parallel for default(none) reduction(+:s) schedule(static) shared (x,L_,kXn,k) private(i) if (nthreads>k)
		// #pragma omp for reduction(+:s) schedule(static)
		for (i = 0; i < k; i++) {
			s += x[i]*L_[kXn+i];
		}
		// #pragma omp single
		x[k] = (b[k] - s) / L_[kXn+k];
	}
	}
	// Backward substitution to solve L'*X = Y; omp does not work
	for (int k = n-1; k >= 0; k--)
	{
		double s = 0.0;
		for (int i = k+1; i < n; i++) {
			s += x[i]*L_[i*n+k];
		}
		x[k] = (x[k] - s)/L_[k*n+k];
	}

	//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	printf(" - chsolve(): %f ms\n",time);

	return;
}

//writes files used for restarting and final output:
void PairRANN::write_potential_file(bool writeparameters, char *header){
	int i,j,k,l;
	FILE *fid = fopen(potential_output_file,"w");
	if (fid==NULL){
		errorf("Invalid parameter file name");
	}
	NNarchitecture *net_out = new NNarchitecture[nelementsp];
	if (normalizeinput){
		unnormalize_net(net_out);
	}
	else {
		copy_network(net,net_out);
	}
	fprintf(fid,"#");
	fprintf(fid,header);
	//atomtypes section
	fprintf(fid,"atomtypes:\n");
	for (i=0;i<nelements;i++){
		fprintf(fid,"%s ",elements[i]);
	}
	fprintf(fid,"\n");
	//mass section
	for (i=0;i<nelements;i++){
		fprintf(fid,"mass:%s:\n",elements[i]);
		fprintf(fid,"%f\n",mass[i]);
	}
	//fingerprints per element section
	for (i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			fprintf(fid,"fingerprintsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",fingerprintperelement[i]);
		}
	}
	//fingerprints section:
	for (i=0;i<nelementsp;i++){
		bool printheader = true;
		for (j=0;j<fingerprintperelement[i];j++){
			if (printheader){
				fprintf(fid,"fingerprints:");
				fprintf(fid,"%s",elementsp[fingerprints[i][j]->atomtypes[0]]);
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					fprintf(fid,"_%s",elementsp[fingerprints[i][j]->atomtypes[k]]);
				}
				fprintf(fid,":\n");
			}
			else {fprintf(fid,"\t");}
			fprintf(fid,"%s_%d",fingerprints[i][j]->style,fingerprints[i][j]->id);
			printheader = true;
			if (j<fingerprintperelement[i]-1 && fingerprints[i][j]->n_body_type == fingerprints[i][j+1]->n_body_type){
				printheader = false;
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					if (fingerprints[i][j]->atomtypes[k]!=fingerprints[i][j+1]->atomtypes[k]){
						printheader = true;
						fprintf(fid,"\n");
						break;
					}
				}
			}
			else fprintf(fid,"\n");
		}
	}
	//fingerprint contants section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<fingerprintperelement[i];j++){
			fingerprints[i][j]->write_values(fid);
		}
	}
	//screening section
	for (i=0;i<nelements;i++){
		for (j=0;j<nelements;j++){
			for (k=0;k<nelements;k++){
				fprintf(fid,"screening:%s_%s_%s:Cmax:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_max[i*nelements*nelements+j*nelements+k]);
				fprintf(fid,"screening:%s_%s_%s:Cmin:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_min[i*nelements*nelements+j*nelements+k]);
			}
		}
	}
	//network layers section:
	for (i=0;i<nelementsp;i++){
		if (net_out[i].layers>0){
			fprintf(fid,"networklayers:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",net_out[i].layers);
		}
	}
	//layer size section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers;j++){
			fprintf(fid,"layersize:%s:%d:\n",elementsp[i],j);
			fprintf(fid,"%d\n",net_out[i].dimensions[j]);
		}
	}
	//weight section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			fprintf(fid,"weight:%s:%d:\n",elementsp[i],j);
			for (k=0;k<net_out[i].dimensions[j+1];k++){
				for (l=0;l<net_out[i].dimensions[j];l++){
					fprintf(fid,"%.15e\t",net_out[i].Weights[j][k*net_out[i].dimensions[j]+l]);
				}
				fprintf(fid,"\n");
			}
		}
	}
	//bias section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			fprintf(fid,"bias:%s:%d:\n",elementsp[i],j);
			for (k=0;k<net_out[i].dimensions[j+1];k++){
				fprintf(fid,"%.15e\n",net_out[i].Biases[j][k]);
			}
		}
	}
	//activation section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			fprintf(fid,"activationfunctions:%s:%d:\n",elementsp[i],j);
			fprintf(fid,"%s\n",activation[i][j]->style);
		}
	}
	//calibration parameters section
	if (writeparameters){
		fprintf(fid,"calibrationparameters:algorithm:\n");
		fprintf(fid,"%s\n",algorithm);
		fprintf(fid,"calibrationparameters:dumpdirectory:\n");
		fprintf(fid,"%s\n",dump_directory);
		fprintf(fid,"calibrationparameters:doforces:\n");
		fprintf(fid,"%d\n",doforces);
		fprintf(fid,"calibrationparameters:normalizeinput:\n");
		fprintf(fid,"%d\n",normalizeinput);
		fprintf(fid,"calibrationparameters:tolerance:\n");
		fprintf(fid,"%.10e\n",tolerance);
		fprintf(fid,"calibrationparameters:regularizer:\n");
		fprintf(fid,"%.10e\n",regularizer);
		fprintf(fid,"calibrationparameters:logfile:\n");
		fprintf(fid,"%s\n",log_file);
		fprintf(fid,"calibrationparameters:potentialoutputfile:\n");
		fprintf(fid,"%s\n",potential_output_file);
		fprintf(fid,"calibrationparameters:potentialoutputfreq:\n");
		fprintf(fid,"%d\n",potential_output_freq);
		fprintf(fid,"calibrationparameters:maxepochs:\n");
		fprintf(fid,"%d\n",max_epochs);
		for (i=0;i<nelementsp;i++){
			for (j=0;j<net_out[i].layers;j++){
				fprintf(fid,"calibrationparameters:dimsreserved:%s:%d:\n",elementsp[i],j);
				fprintf(fid,"%d\n",net_out[i].dimensionsr[j]);
			}
		}
		fprintf(fid,"calibrationparameters:validation:\n");
		fprintf(fid,"%f\n",validation);
	}
	fclose(fid);
	delete [] net_out;
}

void PairRANN::read_atom_types(char **words,char * line1){
	int nwords = 0;
	int t = count_words(line1)+1;
	char **elementword = new char *[t];
	elementword[nwords++] = strtok(line1," ,\t:_\n");
	while ((elementword[nwords++] = strtok(NULL," ,\t:_\n"))) continue;
	if (nwords < 1) errorf("Incorrect syntax for atom types");
	elementword[nwords-1] = new char [strlen("all")+1];
	char elt [] = "all";
	strcpy(elementword[nwords-1],elt);
	nelements = nwords-1;
	allocate(elementword);
}

void PairRANN::read_mass(char **words,char * line1){
//	char str[MAXLINE];
//	sprintf(str,"read_mass\n");
	if (nelements == -1)errorf("atom types must be defined before mass in potential file.");
	int i;
	for (i=0;i<nelements;i++){
		if (strcmp(words[1],elementsp[i])==0){
			mass[i]=strtod(line1,NULL);
			return;
		}
	}
	errorf("mass element not found in atom types.");
}

void PairRANN::read_fpe(char **words,char * line1){
	int i;
	if (nelements == -1)errorf("atom types must be defined before fingerprints per element in potential file.");
	for (i=0;i<nelementsp;i++){
		if (strcmp(words[1],elementsp[i])==0){
			fingerprintperelement[i] = strtol(line1,NULL,10);
			fingerprints[i] = new Fingerprint *[fingerprintperelement[i]];
			for (int j=0;j<fingerprintperelement[i];j++){
				fingerprints[i][j]=new Fingerprint(this);
			}
			return;
		}
	}
	errorf("fingerprint-per-element element not found in atom types");
}

void PairRANN::read_fingerprints(char **words,int nwords,char * line1){
	int nwords1=0,i,j,k,i1;
	bool found;
//	char str[MAXLINE];
	char **words1 = new char * [count_words(line1)+1];
	words1[nwords1++] = strtok(line1," ,\t:_\n");
	while ((words1[nwords1++] = strtok(NULL," ,\t:_\n"))) continue;
	nwords1 -= 1;
	if (nelements == -1)errorf("atom types must be defined before fingerprints in potential file.\n");
	int atomtypes[nwords-1];
	for (i=1;i<nwords;i++){
		found = false;
		for (j=0;j<nelementsp;j++){
			if (strcmp(words[i],elementsp[j])==0){
				atomtypes[i-1]=j;
				found = true;
				break;
			}
		}
		if (!found){errorf("fingerprint element not found in atom types");}
	}
	i = atomtypes[0];
	k = 0;
	if (fingerprintperelement[i]==-1){errorf("fingerprint per element must be defined before fingerprints\n");}
	while (k<nwords1){
		i1 = fingerprintcount[i];
		if (i1>=fingerprintperelement[i]){errorf("more fingerprints found that fingerprintperelement\n");}
//		std::cout<<"got ehre 0\n";
		delete fingerprints[i][i1];
//		std::cout<<"got here 1\n";
		fingerprints[i][i1] = create_fingerprint(words1[k]);
		if (fingerprints[i][i1]->n_body_type!=nwords-1){errorf("invalid fingerprint for element combination\n");}
		k++;
		fingerprints[i][i1]->init(atomtypes,strtol(words1[k++],NULL,10));
		fingerprintcount[i]++;
	}
	delete [] words1;
}


void PairRANN::read_fingerprint_constants(char **words,int nwords,char * line1){
	int i,j,k,i1;
	bool found;
	if (nelements == -1)errorf("atom types must be defined before fingerprints in potential file.");
	int n_body_type = nwords-4;
	int atomtypes[n_body_type];
	for (i=1;i<=n_body_type;i++){
		found = false;
		for (j=0;j<nelementsp;j++){
			if (strcmp(words[i],elementsp[j])==0){
				atomtypes[i-1]=j;
				found = true;
				break;
			}
		}
		if (!found){errorf("fingerprint element not found in atom types");}
	}
	i = atomtypes[0];
	found = false;
	for (k=0;k<fingerprintperelement[i];k++){
		if (fingerprints[i][k]->empty){continue;}
		if (n_body_type!=fingerprints[i][k]->n_body_type){continue;}
		for (j=0;j<n_body_type;j++){
			if (fingerprints[i][k]->atomtypes[j]!=atomtypes[j]){break;}
			if (j==n_body_type-1){
				if (strcmp(words[nwords-3],fingerprints[i][k]->style)==0 && strtol(words[nwords-2],NULL,10)==fingerprints[i][k]->id){
					found=true;
					i1 = k;
					break;
				}
			}
		}
		if (found){break;}
	}
	if (!found){errorf("cannot define constants for unknown fingerprint");}
	fingerprints[i][i1]->fullydefined=fingerprints[i][i1]->parse_values(words[nwords-1],line1);
}

void PairRANN::read_network_layers(char **words,char *line1){
	int i,j;
	if (nelements == -1)errorf("atom types must be defined before network layers in potential file.");
	for (i=0;i<nelements;i++){
		if (strcmp(words[1],elements[i])==0){
			net[i].layers = strtol(line1,NULL,10);
			if (net[i].layers < 1)errorf("invalid number of network layers");
			delete [] net[i].dimensions;
			delete [] net[i].dimensionsr;
			weightdefined[i] = new bool [net[i].layers];
			biasdefined[i] = new bool [net[i].layers];
			net[i].dimensions = new int [net[i].layers];
			net[i].dimensionsr = new int [net[i].layers];
			net[i].Weights = new double * [net[i].layers-1];
			net[i].Biases = new double * [net[i].layers-1];
			net[i].activations = new int [net[i].layers-1];
			for (j=0;j<net[i].layers;j++){
				net[i].dimensions[j]=0;
				net[i].dimensionsr[j]=0;
				if (j<net[i].layers-1)net[i].activations[j]=-1;
				weightdefined[i][j] = false;
				biasdefined[i][j] = false;
			}
			activation[i]=new Activation* [net[i].layers-1];
			for (int j=0;j<net[i].layers-1;j++){
				activation[i][j]= new Activation(this);
			}
			return;
		}
	}
	errorf("network layers element not found in atom types");
}

void PairRANN::read_layer_size(char **words,char* line1){
	int i;
	for (i=0;i<nelements;i++){
		if (strcmp(words[1],elements[i])==0){
			if (net[i].layers==0)errorf("networklayers for each atom type must be defined before the corresponding layer sizes.");
			int j = strtol(words[2],NULL,10);
			if (j>=net[i].layers || j<0){errorf("invalid layer in layer size definition");};
			net[i].dimensions[j]= strtol(line1,NULL,10);
			return;
		}
	}
	errorf("layer size element not found in atom types");
}

void PairRANN::read_weight(char **words,char* line1,FILE* fp){
	int i,j,k,l,nwords;
	char *ptr;
	char **words1;
	for (l=0;l<nelements;l++){
		if (strcmp(words[1],elements[l])==0){
			if (net[l].layers==0)errorf("networklayers must be defined before weights.");
			i=strtol(words[2],NULL,10);
			if (i>=net[l].layers || i<0)errorf("invalid weight layer");
			if (net[l].dimensions[i]==0 || net[l].dimensions[i+1]==0) errorf("network layer sizes must be defined before corresponding weight");
			net[l].Weights[i] = new double [net[l].dimensions[i]*net[l].dimensions[i+1]];
			weightdefined[l][i] = true;
			int n = count_words(line1)+1;
			words1 = new char* [n];
			nwords=0;
			words1[nwords++] = strtok(line1," ,\t:_\n");
			while ((words1[nwords++] = strtok(NULL," ,\t:_\n"))) continue;
			nwords -= 1;
			if (nwords != net[l].dimensions[i])errorf("invalid weights per line");
			for (k=0;k<net[l].dimensions[i];k++){
				net[l].Weights[i][k] = strtod(words1[k],NULL);
			}
			for (j=1;j<net[l].dimensions[i+1];j++){
				ptr = fgets(line1,MAXLINE,fp);
				if (ptr==NULL)errorf("unexpected end of potential file!");
				nwords=0;
				words1[nwords++] = strtok(line1," ,\t:_\n");
				while ((words1[nwords++] = strtok(NULL," ,\t:_\n"))) continue;
				nwords -= 1;
				if (nwords != net[l].dimensions[i])errorf("invalid weights per line");
				for (k=0;k<net[l].dimensions[i];k++){
					net[l].Weights[i][j*net[l].dimensions[i]+k] = strtod(words1[k],NULL);
				}
			}
			delete [] words1;
			return;
		}
	}
	errorf("weight element not found in atom types");
}

void PairRANN::read_bias(char **words,char* line1,FILE* fp){
	int i,j,l;
	char *ptr;
	for (l=0;l<nelements;l++){
		if (strcmp(words[1],elements[l])==0){
			if (net[l].layers==0)errorf("networklayers must be defined before biases.");
			i=strtol(words[2],NULL,10);
			if (i>=net[l].layers || i<0)errorf("invalid bias layer");
			if (net[l].dimensions[i]==0) errorf("network layer sizes must be defined before corresponding bias");
			net[l].Biases[i] = new double [net[l].dimensions[i+1]];
			biasdefined[l][i] = true;
			words[0] = strtok(line1," ,\t:_\n");
			net[l].Biases[i][0] = strtod(words[0],NULL);
			for (j=1;j<net[l].dimensions[i+1];j++){
				ptr = fgets(line1,MAXLINE,fp);
				if (ptr==NULL)errorf("unexpected end of potential file!");
				words[0] = strtok(line1," ,\t:_\n");
				net[l].Biases[i][j] = strtod(words[0],NULL);
			}
			return;
		}
	}
	errorf("bias element not found in atom types");
}

void PairRANN::read_activation_functions(char** words,char * line1){
	int i,l,nwords;
	for (l=0;l<nelements;l++){
		if (strcmp(words[1],elements[l])==0){
			if (net[l].layers==0)errorf("networklayers must be defined before activation functions.");
			i = strtol(words[2],NULL,10);
			if (i>=net[l].layers || i<0)errorf("invalid activation layer");
			nwords=0;
			words[nwords++] = strtok(line1," ,\t:_\n");
			delete activation[l][i];
			activation[l][i]=create_activation(line1);
			return;
		}
	}
	errorf("activation function element not found in atom types");
}

void PairRANN::read_screening(char** words,int nwords,char *line1){
	int i,j,k;
	bool found;
//	char str[MAXLINE];
//	sprintf(str,"%d\n",nwords);
//	for (i=0;i<nwords;i++){
//		std::cout<<words[i];
//		std::cout<<"\n";
//	}
//	std::cout<<str;
	if (nelements == -1)errorf("atom types must be defined before fingerprints in potential file.");
	if (nwords!=5)errorf("invalid screening command");
	int n_body_type = 3;
	int atomtypes[n_body_type];
	for (i=1;i<=n_body_type;i++){
		found = false;
		for (j=0;j<nelementsp;j++){
			if (strcmp(words[i],elementsp[j])==0){
				atomtypes[i-1]=j;
				found = true;
				break;
			}
		}
		if (!found){errorf("fingerprint element not found in atom types");}
	}
	i = atomtypes[0];
	j = atomtypes[1];
	k = atomtypes[2];
	int index = i*nelements*nelements+j*nelements+k;
//	int index1 = i*nelements*nelements+k*nelements+j;
	if (strcmp(words[4],"Cmin")==0)	{
		screening_min[index] = strtod(line1,NULL);
//		screening_min[index1] = screening_min[index];
	}
	else if (strcmp(words[4],"Cmax")==0) {
		screening_max[index] = strtod(line1,NULL);
//		screening_max[index1] = screening_max[index];
	}
	else errorf("unrecognized screening keyword");
}


void PairRANN::screen(double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz, bool *Bij, int ii,int sid,double *xn,double *yn,double *zn,int *tn,int jnum)
{
	//#pragma omp parallel
	{
	//see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
	int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
	double Sijk,Cijk,Cn,Cd,Dij,Dik,Djk,C,dfc,dC,**x;
	PairRANN::Simulation *sim = &sims[sid];
//	x = sim->x;
	double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
	i = sim->ilist[ii];
	itype = map[sim->type[i]];
//	jnum = sim->numneigh[i];
//	jlist = sim->firstneigh[i];
//	xtmp = x[i][0];
//	ytmp = x[i][1];
//	ztmp = x[i][2];
	for (int jj=0;jj<jnum;jj++){
		Sik[jj]=1;
		Bij[jj]=true;
		dSikx[jj]=0;
		dSiky[jj]=0;
		dSikz[jj]=0;
	}
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkx[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijky[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkz[jj*jnum+kk]=0;

	//#pragma omp for schedule (dynamic)for collapse(2)
	for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
		if (Bij[kk]==false){continue;}
//		k = jlist[kk];
//		k &= NEIGHMASK;
//		ktype = map[sim->type[k]];
		ktype = tn[kk];
//		delx2 = xtmp - x[k][0];
//		dely2 = ytmp - x[k][1];
//		delz2 = ztmp - x[k][2];
		delx2 = xn[kk];
		dely2 = yn[kk];
		delz2 = zn[kk];
		rik = delx2*delx2+dely2*dely2+delz2*delz2;
		if (rik>cutmax*cutmax){
			Bij[kk]= false;
			continue;
		}
		for (jj=0;jj<jnum;jj++){
			if (jj==kk){continue;}
			if (Bij[jj]==false){continue;}
//			j = jlist[jj];
//			j &= NEIGHMASK;
//			jtype = map[sim->type[j]];
//			delx = xtmp - x[j][0];
//			dely = ytmp - x[j][1];
//			delz = ztmp - x[j][2];
			jtype = tn[jj];
			delx = xn[jj];
			dely = yn[jj];
			delz = zn[jj];
			rij = delx*delx+dely*dely+delz*delz;
			if (rij>cutmax*cutmax){
				Bij[jj] = false;
				continue;
			}
//			delx3 = x[j][0]-x[k][0];
//			dely3 = x[j][1]-x[k][1];
//			delz3 = x[j][2]-x[k][2];
			delx3 = delx2-delx;
			dely3 = dely2-dely;
			delz3 = delz2-delz;
			rjk = delx3*delx3+dely3*dely3+delz3*delz3;
			if (rik+rjk<=rij){continue;}//bond angle > 90 degrees
			if (rik+rij<=rjk){continue;}//bond angle > 90 degrees
			double Cmax = screening_max[itype*nelements*nelements+jtype*nelements+ktype];
			double Cmin = screening_min[itype*nelements*nelements+jtype*nelements+ktype];
			double temp1 = rij-rik+rjk;
			Cn = temp1*temp1-4*rij*rjk;
			//Cn = (rij-rik+rjk)*(rij-rik+rjk)-4*rij*rjk;
			temp1 = rij-rjk;
			Cd = temp1*temp1-rik*rik;
			//Cd = (rij-rjk)*(rij-rjk)-rik*rik;
			Cijk = Cn/Cd;
			//Cijk = 1+2*(rik*rij+rik*rjk-rik*rik)/(rik*rik-(rij-rjk)*(rij-rjk));
			C = (Cijk-Cmin)/(Cmax-Cmin);
			if (C>=1){continue;}
			else if (C<=0){
				Bij[kk]=false;
				break;
			}
			dC = Cmax-Cmin;
			dC *= dC;
			dC *= dC;
			temp1 = 1-C;
			temp1 *= temp1;
			temp1 *= temp1;
			Sijk = 1-temp1;
			Sijk *= Sijk;
			Dij = 4*rik*(Cn+4*rjk*(rij+rik-rjk))/Cd/Cd;
			Dik = -4*(rij*Cn+rjk*Cn+8*rij*rik*rjk)/Cd/Cd;
			Djk = 4*rik*(Cn+4*rij*(rik-rij+rjk))/Cd/Cd;
			temp1 = Cijk-Cmax;
			double temp2 = temp1*temp1;
			dfc = 8*temp1*temp2/(temp2*temp2-dC);
			Sik[kk] *= Sijk;
			dSijkx[kk*jnum+jj] = dfc*(delx*Dij-delx3*Djk);
			dSikx[kk] += dfc*(delx2*Dik+delx3*Djk);
			dSijky[kk*jnum+jj] = dfc*(dely*Dij-dely3*Djk);
			dSiky[kk] += dfc*(dely2*Dik+dely3*Djk);
			dSijkz[kk*jnum+jj] = dfc*(delz*Dij-delz3*Djk);
			dSikz[kk] += dfc*(delz2*Dik+delz3*Djk);
		}
	}
	}
}

//treats # as starting a comment to be ignored.
int PairRANN::count_words(char *line){
	int n = strlen(line) + 1;
	char copy[n];
	strncpy(copy,line,n);
	char *ptr;
	if ((ptr = strchr(copy,'#'))) *ptr = '\0';
	if (strtok(copy," ,\t:_\n") == NULL) {
		return 0;
	}
	n=1;
	while ((strtok(NULL," ,\t:_\n"))) n++;
	return n;
}


void PairRANN::errorf(const char *message){
	//see about adding message to log file
	std::cout<<message;
	exit(1);
}

template <typename T>
Fingerprint *PairRANN::fingerprint_creator(PairRANN* pair)
{
  return new T(pair);
}

Fingerprint *PairRANN::create_fingerprint(const char *style)
{
	if (fingerprint_map->find(style) != fingerprint_map->end()) {
		FingerprintCreator fingerprint_creator = (*fingerprint_map)[style];
		return fingerprint_creator(this);
	}
	char str[128];
	sprintf(str,"Unknown fingerprint style %s",style);
	errorf(str);
	return NULL;
}

template <typename T>
Activation *PairRANN::activation_creator(PairRANN* pair)
{
  return new T(pair);
}

Activation *PairRANN::create_activation(const char *style)
{
	if (activation_map->find(style) != activation_map->end()) {
		ActivationCreator activation_creator = (*activation_map)[style];
		return activation_creator(this);
	}
	char str[128];
	sprintf(str,"Unknown activation style %s",style);
	errorf(str);
	return NULL;
}
