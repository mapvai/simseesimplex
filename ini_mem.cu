#include <stdio.h>
#include <math.h>

#include "toursimplexmgpus_final.h"


extern "C" void ini_mem(
	TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, 
	int NTrayectorias, int NEnteras, int NVariables, int NRestricciones, int cnt_varfijas, int cnt_RestriccionesRedundantes
) {
	
	int largo, alto;
	
	h_simplex_array = (TabloideGPUs*)malloc(NTrayectorias*sizeof(TabloideGPUs));

	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		
		largo = (int) h_simplex_array[kTrayectoria][3];
		alto = (int) h_simplex_array[kTrayectoria][largo + 1] + 6;
		cudaMalloc(&h_simplex_array[kTrayectoria], largo*alto*sizeof(double));
	
	}

 	cudaMalloc(&d_simplex_array, NTrayectorias*sizeof(TabloideGPUs));
	cudaMemcpy(d_simplex_array, h_simplex_array, NTrayectorias*sizeof(TabloideGPUs), cudaMemcpyHostToDevice);
 
}
