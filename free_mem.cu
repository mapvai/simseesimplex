#include <stdio.h>
#include <math.h>

#include "toursimplexmgpus_final.h"


extern "C" void free_mem(
	TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, 
	int NTrayectorias, int NEnteras, int NVariables, int NRestricciones, int cnt_varfijas, int cnt_RestriccionesRedundantes
						
) {
	
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		cudaFree(h_simplex_array[kTrayectoria]);
	}
	
	cudaFree(d_simplex_array);
 
}
