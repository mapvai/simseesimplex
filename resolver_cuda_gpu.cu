#include <stdio.h>
#include <math.h>
#include "tsimplexgpus.h"

#define BLOCK_SIZE_P_I 64 //  = 2 * WARP SIZE
#define BLOCK_SIZE_GR 160 //  = 5 * WARP SIZE

const double CasiCero_Simplex = 1.0E-7;
const double CasiCero_Simplex_CotaSup = CasiCero_Simplex * 1.0E+3;
const double CasiCero_CajaLaminar = 1.0E-30;
const double AsumaCero =  1.0E-16; // EPSILON de la maquina en cuentas con Double, CONFIRMAR SI ESTO ES CORRECTO (double 64 bits, 11 exponente y 53 mantisa, 53 log10(2) ≈ 15.955 => 2E−53 ≈ 1.11 × 10E−16 => EPSILON  ≈ 1.0E-16)
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

__device__ void posicionarPrimeraLibre(TSimplexGPUs &smp, int cnt_varfijas, int &cnt_fijadas, int &cnt_columnasFijadas, int &kPrimeraLibre) ; // Esta funcion es interna, no se necesita declarar aca?
__device__ bool fijarVariables(TSimplexGPUs &smp, int cnt_varfijas, int &cnt_columnasFijadas, int cnt_RestriccionesRedundantes);
__device__ void intercambiar(TSimplexGPUs &smp, int kfil, int jcol);
__device__ void intercambiar_multihilo(TSimplexGPUs &smp, int kfil, int jcol);
__device__ void intercambioColumnas(TSimplexGPUs &smp, int j1, int j2);
__device__ void intercambioFilas(TSimplexGPUs &smp, int k1, int k2);
__device__ int buscarMejorPivoteEnCol(TSimplexGPUs &smp, int jCol, int iFilaFrom, int iFilaHasta);
__device__ bool enfilarVariablesLibres(TSimplexGPUs &smp, int cnt_columnasFijadas, int &cnt_RestriccionesRedundantes, int &cnt_variablesLiberadas);
__device__ int resolverIgualdades(TSimplexGPUs &smp, int &cnt_columnasFijadas, int cnt_varfijas, int &cnt_RestriccionesRedundantes, int cnt_igualdades);
__device__ bool filaEsFactible(TSimplexGPUs &smp, int kfila, bool &fantasma);
__device__ int pasoBuscarFactibleIgualdad4(TSimplexGPUs &smp, int IgualdadesNoResueltas, int * nCerosFilas, int * nCerosCols, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes);
__device__ int reordenarPorFactibilidad(TSimplexGPUs &smp, int cnt_RestriccionesRedundantes, int &cnt_RestrInfactibles);
__device__ void cambiar_borde_de_caja(TSimplexGPUs &smp, int k_fila);
__device__ int pasoBuscarFactible(TSimplexGPUs &smp, int &cnt_RestrInfactibles, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes);
__device__ int locate_zpos(TSimplexGPUs &smp, int kfila_z, int cnt_columnasFijadas);
__device__ void capturarElMejor(double &a_iq_DelMejor, double &a_it_DelMejor, int &p, bool &filaFantasma, bool &colFantasma, double a_iq, double a_it, int i, bool xFantasma_fila); // Esta funcion es interna, no se necesita declarar aca?
__device__ int mejorpivote(TSimplexGPUs &smp, int q, int kmax, bool &filaFantasma, bool &colFantasma, bool checkearFilaOpt, int cnt_RestriccionesRedundantes);
__device__ bool cambio_var_cota_sup_en_columna(TSimplexGPUs &smp, int q);
__device__ int locate_qOK(TSimplexGPUs &smp, int p, int jhasta, int jti, int cnt_RestriccionesRedundantes);
__device__ bool test_qOK(TSimplexGPUs &smp, int p, int q, int jti, double &apq, int cnt_RestriccionesRedundantes);
__device__ void darpaso(TSimplexGPUs &smp, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes, int &res);

__device__ void resolver_etapa_cnt_igcount(TSimplexGPUs &simplex,  TSimplexVars &d_svars) ;
__device__ void resolver_etapa_fijar_cajlam(TSimplexGPUs &simplex,  TSimplexVars &d_svars) ;
__device__ void resolver_fijar_variables(TSimplexGPUs &simplex, TSimplexVars &d_svars);
__device__ void resolver_enfilar_variables_libres(TSimplexGPUs &simplex, TSimplexVars &d_svars);
__device__ void resolver_igualdades(TSimplexGPUs &simplex, TSimplexVars &d_svars);
__device__ void resolver_reordenar_por_factibilidad(TSimplexGPUs &simplex, TSimplexVars &d_svars);
__device__ void resolver_paso_iterativo(TSimplexGPUs &simplex, TSimplexVars &d_svars) ;

__global__ void kernel_resolver_etapa_cnt_igcount(TDAOfSimplexGPUs simplex_array,  TSimplexVars * d_svars) {
	resolver_etapa_cnt_igcount(simplex_array[blockIdx.x], d_svars[blockIdx.x]);
} 

__global__ void kernel_resolver_etapa_fijar_cajlam(TDAOfSimplexGPUs simplex_array,  TSimplexVars * d_svars) {	
	resolver_etapa_fijar_cajlam(simplex_array[blockIdx.x], d_svars[blockIdx.x]);
}

__global__ void kernel_resolver_fijar_variables(TDAOfSimplexGPUs simplex_array, TSimplexVars * d_svars, int NTrayectorias) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < NTrayectorias) resolver_fijar_variables(simplex_array[index], d_svars[index]);
}

__global__ void kernel_resolver_enfilar_variables_libres(TDAOfSimplexGPUs simplex_array, TSimplexVars * d_svars, int NTrayectorias) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < NTrayectorias) resolver_enfilar_variables_libres(simplex_array[index], d_svars[index]);
}

__global__ void kernel_resolver_igualdades(TDAOfSimplexGPUs simplex_array, TSimplexVars * d_svars, int NTrayectorias) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < NTrayectorias) resolver_igualdades(simplex_array[index], d_svars[index]);
}

__global__ void kernel_resolver_reordenar_por_factibilidad(TDAOfSimplexGPUs simplex_array, TSimplexVars * d_svars, int NTrayectorias) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < NTrayectorias) resolver_reordenar_por_factibilidad(simplex_array[index], d_svars[index]);
}

__global__ void kernel_resolver_paso_iterativo(TDAOfSimplexGPUs simplex_array, TSimplexVars * d_svars) {
	resolver_paso_iterativo(simplex_array[blockIdx.x], d_svars[blockIdx.x]);
}

extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	int NEnteras;
	int NVariables;
	int NRestricciones;
	int cnt_varfijas;
	int cnt_RestriccionesRedundantes;
	TSimplexVars * d_svars;
	
	int maxNvariables = 0;
	int maxNrestricciones = 0;
	
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		NEnteras = simplex_array[kTrayectoria].NEnteras;
		NVariables = simplex_array[kTrayectoria].NVariables;
		NRestricciones = simplex_array[kTrayectoria].NRestricciones;
		cnt_varfijas = simplex_array[kTrayectoria].cnt_varfijas;
		cnt_RestriccionesRedundantes = simplex_array[kTrayectoria].cnt_RestriccionesRedundantes;
		h_simplex_array[kTrayectoria].NEnteras = NEnteras;
		h_simplex_array[kTrayectoria].NVariables = NVariables;	
		h_simplex_array[kTrayectoria].NRestricciones = NRestricciones;		
		h_simplex_array[kTrayectoria].cnt_varfijas = cnt_varfijas;
		h_simplex_array[kTrayectoria].cnt_RestriccionesRedundantes = cnt_RestriccionesRedundantes;
		
		if (NVariables > maxNvariables) maxNvariables = NVariables;
		if (NRestricciones > maxNrestricciones) maxNrestricciones = NRestricciones;

		cudaMemcpy(h_simplex_array[kTrayectoria].x_inf, simplex_array[kTrayectoria].x_inf, NVariables*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].x_sup, simplex_array[kTrayectoria].x_sup, NVariables*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].flg_x, simplex_array[kTrayectoria].flg_x, NVariables*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].top, simplex_array[kTrayectoria].top, NVariables*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].flg_y, simplex_array[kTrayectoria].flg_y, NRestricciones*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].left, simplex_array[kTrayectoria].left, NRestricciones*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(h_simplex_array[kTrayectoria].lstvents, simplex_array[kTrayectoria].lstvents, NEnteras*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_simplex_array[kTrayectoria].lstAcoplesVEnts, simplex_array[kTrayectoria].lstAcoplesVEnts, 2*NEnteras*sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(h_simplex_array[kTrayectoria].mat, simplex_array[kTrayectoria].mat, (NVariables + 1)*(NRestricciones + 1)*sizeof(double), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(d_simplex_array, h_simplex_array, NTrayectorias*sizeof(TSimplexGPUs), cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_svars, NTrayectorias*sizeof(TSimplexVars));
	cudaMemset(d_svars, 0, NTrayectorias*sizeof(TSimplexVars));
	
	cudaError_t err;
	
	// Ejecuto los kernels
	const dim3 DimGrid_e1(NTrayectorias, 1);
	const dim3 DimBlock_e1(maxNrestricciones, 1);
	kernel_resolver_etapa_cnt_igcount<<< DimGrid_e1, DimBlock_e1, 0, 0 >>>(d_simplex_array, d_svars);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 1 error", cudaGetErrorString(err));
	
	const dim3 DimGrid_e2(NTrayectorias, 1);
	const dim3 DimBlock_e2(maxNvariables, 1);
	kernel_resolver_etapa_fijar_cajlam<<< DimGrid_e2, DimBlock_e2, 0, 0 >>>(d_simplex_array, d_svars);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 2 error", cudaGetErrorString(err));
	
	
	int cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	printf("%s: %d\n", "cantBloques fijar_variables: ", cantBloques);
	const dim3 DimGrid_e3(cantBloques, 1);
	const dim3 DimBlock_e3(BLOCK_SIZE_GR, 1);
	kernel_resolver_fijar_variables<<< DimGrid_e3, DimBlock_e3, 0, 0 >>>(d_simplex_array, d_svars, NTrayectorias);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 3 error", cudaGetErrorString(err));
	
	
	cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	printf("%s: %d\n", "cantBloques enfilar_variables_libres: ", cantBloques);
	const dim3 DimGrid_e4(cantBloques, 1);
	const dim3 DimBlock_e4(BLOCK_SIZE_GR, 1);
	kernel_resolver_enfilar_variables_libres<<< DimGrid_e4, DimBlock_e4, 0, 0 >>>(d_simplex_array, d_svars, NTrayectorias);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 4 error", cudaGetErrorString(err));
	
	cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	printf("%s: %d\n", "cantBloques resolver_igualdades: ", cantBloques);
	const dim3 DimGrid_e5(cantBloques, 1);
	const dim3 DimBlock_e5(BLOCK_SIZE_GR, 1);
	kernel_resolver_igualdades<<< DimGrid_e5, DimBlock_e5, 0, 0 >>>(d_simplex_array, d_svars, NTrayectorias);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 5 error", cudaGetErrorString(err));
	
	
	cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	printf("%s: %d\n", "cantBloques reordenar_por_factibilidad: ", cantBloques);
	const dim3 DimGrid_e6(cantBloques, 1);
	const dim3 DimBlock_e6(BLOCK_SIZE_GR, 1);
	kernel_resolver_reordenar_por_factibilidad<<< DimGrid_e6, DimBlock_e6, 0, 0 >>>(d_simplex_array, d_svars, NTrayectorias);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 6 error", cudaGetErrorString(err));
	
	const dim3 DimGrid_e7(NTrayectorias, 1);
	const dim3 DimBlock_e7(BLOCK_SIZE_P_I, 1);
	kernel_resolver_paso_iterativo<<< DimGrid_e7, DimBlock_e7, 0, 0 >>>(d_simplex_array, d_svars);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 7 error", cudaGetErrorString(err));

	cudaMemcpy(h_simplex_array, d_simplex_array, NTrayectorias*sizeof(TSimplexGPUs), cudaMemcpyDeviceToHost);

	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		int NEnteras = h_simplex_array[kTrayectoria].NEnteras;
		int NVariables = h_simplex_array[kTrayectoria].NVariables;
		int NRestricciones = h_simplex_array[kTrayectoria].NRestricciones;
		int cnt_varfijas = h_simplex_array[kTrayectoria].cnt_varfijas;
		int cnt_RestriccionesRedundantes = h_simplex_array[kTrayectoria].cnt_RestriccionesRedundantes;

		simplex_array[kTrayectoria].NEnteras = NEnteras;
		simplex_array[kTrayectoria].NVariables = NVariables;	
		simplex_array[kTrayectoria].NRestricciones = NRestricciones;		
		simplex_array[kTrayectoria].cnt_varfijas = cnt_varfijas;
		simplex_array[kTrayectoria].cnt_RestriccionesRedundantes = cnt_RestriccionesRedundantes;
		cudaMemcpy(simplex_array[kTrayectoria].x_inf, h_simplex_array[kTrayectoria].x_inf, NVariables*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].x_sup, h_simplex_array[kTrayectoria].x_sup, NVariables*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].flg_x, h_simplex_array[kTrayectoria].flg_x, NVariables*sizeof(char), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].top,   h_simplex_array[kTrayectoria].top, NVariables*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].flg_y, h_simplex_array[kTrayectoria].flg_y, NRestricciones*sizeof(char), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].left,  h_simplex_array[kTrayectoria].left, NRestricciones*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].lstvents,  h_simplex_array[kTrayectoria].lstvents, NEnteras*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].lstAcoplesVEnts, h_simplex_array[kTrayectoria].lstAcoplesVEnts, 2*NEnteras*sizeof(char), cudaMemcpyDeviceToHost);
		cudaMemcpy(simplex_array[kTrayectoria].mat, h_simplex_array[kTrayectoria].mat, (NVariables + 1)*(NRestricciones + 1)*sizeof(double), cudaMemcpyDeviceToHost);
	}	

}

/**************************************************************************************************************************************************************************************************/

__device__ void resolver_etapa_cnt_igcount(TSimplexGPUs &simplex,  TSimplexVars &d_svars) {

	__shared__ int cnt_igualdades;
	
	if (threadIdx.x == 0)  {
		cnt_igualdades = 0;
		d_svars.res = 1;
	}
	
	__syncthreads();
	
	for (int i = threadIdx.x; i <= simplex.NRestricciones; i += blockDim.x) {
		if (simplex.flg_y[i] == 2) {
			atomicAdd(&cnt_igualdades, 1);
		}
	}
	
	__syncthreads();
	
	if (threadIdx.x == 0)  {
		d_svars.cnt_igualdades = cnt_igualdades;
	}
}

// fijarCajasLaminares()
__device__ void resolver_etapa_fijar_cajlam(TSimplexGPUs &simplex,  TSimplexVars &d_svars) {
	// simplex.cnt_varfijas = 0; // Esto viene al cargarse la estructura
	// Si abs( cotasup - cotainf ) < AsumaCeroCaja then flg_x := 2
	// Retorna la cantidad de cajas fijadas.
	for (int i = threadIdx.x; i < simplex.NVariables; i += blockDim.x) {
		if (abs(simplex.flg_x[i] ) < 2) { // No se aplica ni para 2 ni para 3
			if (abs(simplex.x_sup[i] - simplex.x_inf[i] ) < CasiCero_CajaLaminar) {
				if (simplex.flg_x[i] < 0) {
					simplex.flg_x[i] = -2;
				} else {
					simplex.flg_x[i] = 2;
				}
				atomicAdd(&simplex.cnt_varfijas, 1);
			}
		}
	}
}

__device__ void resolver_fijar_variables(TSimplexGPUs &simplex, TSimplexVars &d_svars) {

	// Fijamos las variables que se hayan declarado como constantes.
	if (!fijarVariables(simplex, simplex.cnt_varfijas, d_svars.cnt_columnasFijadas, simplex.cnt_RestriccionesRedundantes)) {
		printf("%s\n", "PROBLEMA INFACTIBLE - No fijar las variable Fijas.");
		d_svars.res = -32;
	}
	
}

__device__ void resolver_enfilar_variables_libres(TSimplexGPUs &simplex, TSimplexVars &d_svars) {
	
	if (d_svars.res != 1) return;
	
	if (!enfilarVariablesLibres(simplex, d_svars.cnt_columnasFijadas, simplex.cnt_RestriccionesRedundantes, d_svars.cnt_variablesLiberadas)) {
		printf("%s\n", "No fue posible conmutar a filas todas las variables libres");
		d_svars.res = -33;
	}
	
}

__device__ void resolver_igualdades(TSimplexGPUs &simplex, TSimplexVars &d_svars) {
	
	if (d_svars.res != 1) return;
	
	if (resolverIgualdades(simplex, d_svars.cnt_columnasFijadas, simplex.cnt_varfijas, simplex.cnt_RestriccionesRedundantes, d_svars.cnt_igualdades) != 1) {
		printf("%s\n", "No fue posible conmutar a filas todas las variables libres");
		d_svars.res = -33;
	}
	
}

__device__ void resolver_reordenar_por_factibilidad(TSimplexGPUs &simplex, TSimplexVars &d_svars) {
	
	reordenarPorFactibilidad(simplex, simplex.cnt_RestriccionesRedundantes, d_svars.cnt_RestrInfactibles); // MAP: cnt_RestrInfactibles es modificada dentro del proc
	
}

__device__ void resolver_paso_iterativo(TSimplexGPUs &simplex, TSimplexVars &d_svars) {
	__shared__ int res;
	__shared__ int cnt_columnasFijadas; // Cantidad de columnas FIJADAS x o y fija encolumnada
	__shared__ int cnt_RestrInfactibles;
	
	if (d_svars.res != 1) return;
	
	if (threadIdx.x == 0)  {
		cnt_columnasFijadas = d_svars.cnt_columnasFijadas;
		cnt_RestrInfactibles = d_svars.cnt_RestrInfactibles;
	}
	
	lbl_inicio:
	
	__syncthreads();
	
	if (threadIdx.x == 0)  {
		
		while (cnt_RestrInfactibles > 0) {
			res = pasoBuscarFactible(simplex, cnt_RestrInfactibles, cnt_columnasFijadas, simplex.cnt_RestriccionesRedundantes);

			switch (res) {
				case 0: 
					if (cnt_RestrInfactibles > 0) {
						printf("%s\n", "PROBLEMA INFACTIBLE - Buscando factibilidad");
						res = -10;
						d_svars.res = res;
						return;
					}
					break;
				case  -1:
					printf("%s\n", "NO encontramos pivote bueno - Buscando Factibilidad");
					res = -11;
					d_svars.res = res;
					return;
				case -2:
					printf("%s\n", "???cnt_infactibles= 0 - Buscando Factibilidad");
					res = -12;
					d_svars.res = res;
					return;
			}
		}
	}
	
	__syncthreads();

	while (res == 1) {
		darpaso(simplex, cnt_columnasFijadas, simplex.cnt_RestriccionesRedundantes, res);
		__syncthreads();
		if (res == -1) {
			if (threadIdx.x == 0) printf("%s\n", "Error -- NO encontramos pivote bueno dando paso");
			if (threadIdx.x == 0)  d_svars.res = -21;
			return;
		}
	}
	  
	if (res == 2) {
		//  Trabaja solo el primer hilo
		if (threadIdx.x == 0)  {
			reordenarPorFactibilidad(simplex, simplex.cnt_RestriccionesRedundantes, cnt_RestrInfactibles); // MAP: cnt_RestrInfactibles es modificada dentro del proc
			res = 1;
		}
		goto lbl_inicio;
	}
	
	if (threadIdx.x == 0)  d_svars.res = res;

	// printf("%s: %d\n", "Finish, result = ", res);

}

__device__ void posicionarPrimeraLibre(TSimplexGPUs &smp, int cnt_varfijas, int &cnt_fijadas, int &cnt_columnasFijadas, int &kPrimeraLibre) {
    while ((cnt_fijadas < cnt_varfijas) && 
		(((smp.top[kPrimeraLibre] < 0) && (abs(smp.flg_x[-smp.top[kPrimeraLibre]]) == 2)) || ((smp.top[kPrimeraLibre] > 0) && (abs(smp.flg_y[smp.top[kPrimeraLibre]]) == 2)))) {
		if (smp.top[kPrimeraLibre] < 0) {
			cnt_fijadas++;
		}	
		cnt_columnasFijadas++;
		kPrimeraLibre--;
	}
}


__device__ bool fijarVariables(TSimplexGPUs &smp, int cnt_varfijas, int &cnt_columnasFijadas, int cnt_RestriccionesRedundantes) {

	int kColumnas, mejorColumnaParaCambiarFila, kFor, kFilaAFijar, cnt_fijadas, kPrimeraLibre;
	double mejorAkFilai;
	bool buscando;

	if (cnt_varfijas > 0) {
		cnt_fijadas = 0;
		kPrimeraLibre = smp.NVariables - 1; // MAP: nc - 1, - 1 debido al cambio de indices
		kColumnas = 0; // MAP: Antes 1, pero cambimos que los vectores esten indexados desde 1 a 0
		
		while ((cnt_fijadas < cnt_varfijas) && (kColumnas <= kPrimeraLibre)) {
			posicionarPrimeraLibre(smp, cnt_varfijas, cnt_fijadas, cnt_columnasFijadas, kPrimeraLibre);
			//Busco en columnas
			if ((cnt_fijadas < cnt_varfijas) && (kColumnas <= kPrimeraLibre)) {
				buscando = true;
				while (buscando && (kColumnas <= kPrimeraLibre)) {
					
					// MAP: Aca grego -1 a flg_x[-top[kColumnas] -1] para obtener el indice que empieza en 0 y top left son 2 vectores indexados en 0 pero que contienen numeros comenzados en 1, por lo tanto el indice real es top[indice] - 1,
					// esto lo voy a tener que hacer a lo largo y ancho del algoritmo
					if ((smp.top[kColumnas] < 0) && (abs(smp.flg_x[-smp.top[kColumnas] - 1]) == 2)) { 
						// Es una x fija
						buscando = false;
					} else {
						kColumnas++;
					}
				}
				if (!buscando) {
					intercambioColumnas(smp, kColumnas, kPrimeraLibre);
					kPrimeraLibre--;
					cnt_fijadas++;
					cnt_columnasFijadas++;
					kColumnas++;
				}
			}
		}

		// Se inicializa en la fila anterior a la ultima fijada
		kFilaAFijar = cnt_RestriccionesRedundantes - 1; // MAP: Agreo -1
		while (cnt_fijadas < cnt_varfijas) {
		
			posicionarPrimeraLibre(smp, cnt_varfijas, cnt_fijadas, cnt_columnasFijadas, kPrimeraLibre);
			if (cnt_fijadas < cnt_varfijas) {
		 
				//Busco en filas
				for (kFor = kFilaAFijar + 1; kFor < smp.NVariables - 1; kFor++) { // MAP: change index in loop
					if ((smp.left[kFor] < 0) && (abs(smp.flg_x[-smp.left[kFor] - 1]) == 2)) {
						kFilaAFijar = kFor;
						break;
					}
				}

				mejorColumnaParaCambiarFila = 0; // MAP: antes 1
				mejorAkFilai = abs(smp.mat[kFilaAFijar*(smp.NVariables + 1)]); // MAP: before [1], primera columna ahora es 0, [kFilaAFijar][0]
				for (kColumnas = 1; kColumnas < kPrimeraLibre; kColumnas++) { // MAP: Antes 2 to kPrimeraLibre
					if (abs(smp.mat[kFilaAFijar * (smp.NVariables + 1) + kColumnas]) > mejorAkFilai) {
						mejorColumnaParaCambiarFila = kColumnas;
						mejorAkFilai = abs(smp.mat[kFilaAFijar * (smp.NVariables + 1) + kColumnas]);
					}
				}

				// dv@20191226 Si el término independiente es nulo, la "restricción" es reduntante
				// entonces la variable ya había quedado fijada
				if (mejorAkFilai < AsumaCero) {
					printf("%s: %g, %g\n", "fijarVariables false  mejorAkFilai < AsumaCero: ", mejorAkFilai, AsumaCero);
					return false;
				}

				intercambiar(smp, kFilaAFijar, mejorColumnaParaCambiarFila);
				cnt_fijadas++;
				
				// dv@20200115 agrego esto porque no debería cambiar si pivoteó con una fijada
				if (mejorColumnaParaCambiarFila != kPrimeraLibre) {
					intercambioColumnas(smp, mejorColumnaParaCambiarFila, kPrimeraLibre);
				}
				cnt_columnasFijadas++;
				kPrimeraLibre--;
			}
		}
	}
	
	return true;

}


__device__ void intercambiar(TSimplexGPUs &smp, int kfil, int jcol) {

	double m, piv, invPiv;
	int k, j;
	
	// printf("%s\n", "INTERCAMBIAR AAAAAAAAAAAAAAA");

	piv = smp.mat[kfil * (smp.NVariables + 1) + jcol];
	invPiv = 1 / piv;

	// MAP: POSIBLE MEJORA, en GPU se encargaran hilos diferentes, recorrida por filas
	// MAP: En el codigo original se separa en 4 casos el mismo codigo, se recorre desde 1 to cnt_RestriccionesRedundantes to kfil - 1  <skipping kfil (fila pivot)> to nf - 1 to nf
	// MAP: Se concatenan estos casos en un mismo for
	for (k = 0; k < kfil; k++) {
		m = -smp.mat[k * (smp.NVariables + 1) + jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j]; 
			}
			
			smp.mat[k * (smp.NVariables + 1) + jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j];
			}
		} else {
			smp.mat[k * (smp.NVariables + 1) + jcol] = 0; // IMPONGO_CEROS
		}
	}
	
	// Salteo la fila kfil
	
	for (k = kfil + 1; k <= smp.NRestricciones; k++) { // MAP: <= pq considera la fila de la funcion z
		m = -smp.mat[k * (smp.NVariables + 1) + jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j]; 
			}
			
			smp.mat[k * (smp.NVariables + 1) + jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j];
			}
		} else {
			smp.mat[k * (smp.NVariables + 1) + jcol] = 0; // IMPONGO_CEROS
		}
	}

	// Completo la fila kfil
	m = -invPiv;
	for (j = 0; j < jcol; j++) {
		smp.mat[kfil * (smp.NVariables + 1) + j] = smp.mat[kfil * (smp.NVariables + 1) + j] * m;
	}

	smp.mat[kfil * (smp.NVariables + 1) + jcol] = -m;
	
	for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes jcol + 1 to nc
		smp.mat[kfil * (smp.NVariables + 1) + j] = smp.mat[kfil * (smp.NVariables + 1) + j] * m;
	}

	k = smp.top[jcol];
	smp.top[jcol] = smp.left[kfil];
	smp.left[kfil] = k;

 }
 
 __device__ void intercambiar_multihilo(TSimplexGPUs &smp, int kfil, int jcol) {

	double m, piv, invPiv;
	int k, j;
	
	// printf("%s\n", "INTERCAMBIAR MULTIHILO AAAAAAAAAAAAAAA");

	piv = smp.mat[kfil * (smp.NVariables + 1) + jcol];
	invPiv = 1 / piv;

	// MAP: POSIBLE MEJORA, en GPU se encargaran hilos diferentes, recorrida por filas
	// MAP: En el codigo original se separa en 4 casos el mismo codigo, se recorre desde 1 to cnt_RestriccionesRedundantes to kfil - 1  <skipping kfil (fila pivot)> to nf - 1 to nf
	// MAP: Se concatenan estos casos en un mismo for
	for (k = threadIdx.x; k < kfil; k+= blockDim.x) {
		m = -smp.mat[k * (smp.NVariables + 1) + jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j]; 
			}
			
			smp.mat[k * (smp.NVariables + 1) + jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j];
			}
		} else {
			smp.mat[k * (smp.NVariables + 1) + jcol] = 0; // IMPONGO_CEROS
		}
	}
	
	// Salteo la fila kfil
	
	for (k = kfil + 1 + threadIdx.x; k <= smp.NRestricciones; k += blockDim.x) { // MAP: <= pq considera la fila de la funcion z
		m = -smp.mat[k * (smp.NVariables + 1) + jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j]; 
			}
			
			smp.mat[k * (smp.NVariables + 1) + jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k * (smp.NVariables + 1) + j] = smp.mat[k * (smp.NVariables + 1) + j] + m * smp.mat[kfil * (smp.NVariables + 1) + j];
			}
		} else {
			smp.mat[k * (smp.NVariables + 1) + jcol] = 0; // IMPONGO_CEROS
		}
	}

	// Completo la fila kfil
	for (j = threadIdx.x; j < jcol; j += blockDim.x) {
		smp.mat[kfil * (smp.NVariables + 1) + j] = -smp.mat[kfil * (smp.NVariables + 1) + j] * invPiv;
	}

	if (threadIdx.x == 0) smp.mat[kfil * (smp.NVariables + 1) + jcol] = invPiv;
	
	for (j = jcol + 1 + threadIdx.x; j <= smp.NVariables; j += blockDim.x) { // MAP: Antes jcol + 1 to nc
		smp.mat[kfil * (smp.NVariables + 1) + j] = -smp.mat[kfil * (smp.NVariables + 1) + j] * invPiv;
	}
	
	if (threadIdx.x == 0) {
		k = smp.top[jcol];
		smp.top[jcol] = smp.left[kfil];
		smp.left[kfil] = k;
	}

 }

__device__ void intercambioColumnas(TSimplexGPUs &smp, int j1, int j2) {

	int k;
	double m;

	for (k = 0; k <= smp.NRestricciones; k++) { // MAP: Intercambia toda la columna por lo tanto va hasta smp.NRestricciones inclusive
		m = smp.mat[k * (smp.NVariables + 1) + j1];
		smp.mat[k * (smp.NVariables + 1) + j1]  = smp.mat[k * (smp.NVariables + 1) + j2];
		smp.mat[k * (smp.NVariables + 1) + j2]  = m;
	}

	k = smp.top[j1];
	smp.top[j1] = smp.top[j2];
	smp.top[j2] = k;

}

__device__ void intercambioFilas(TSimplexGPUs &smp, int k1, int k2) {
  
	int j;
	double m;

	for (j = 0; j <= smp.NVariables; j++) { // MAP: Intercambia toda la fila por lo tanto va hasta smp.NVariables inclusive
		m = smp.mat[k1 * (smp.NVariables + 1) + j];
		smp.mat[k1 * (smp.NVariables + 1) + j] = smp.mat[k2 * (smp.NVariables + 1) + j];
		smp.mat[k2 * (smp.NVariables + 1) + j] = m;
	}

	j = smp.left[k1];
	smp.left[k1] = smp.left[k2];
	smp.left[k2] = j;

}

 // Atención esto funciona porque suponemos que el Simplex está en su estado Natural.
 // a lo sumo se conmutaron algunas columnas para fijar variables.
 // MAP: se usa solo dentro de enfilarVariablesLibres
__device__ int buscarMejorPivoteEnCol(TSimplexGPUs &smp, int jCol, int iFilaFrom, int iFilaHasta) {
	
	double a;
	int iFil, res;
	
    res = -1;
    a = 0.0;
	for (iFil = iFilaFrom; iFil <= iFilaHasta; iFil++) { // MAP: Originalmente iFilaFrom to iFilaHasta, eso no cambia ya que se le pasan los indices ya adaptados a la indexacion desde 0
		if (abs(smp.mat[iFil * (smp.NVariables + 1) + jCol]) > a) {
			a = abs(smp.mat[iFil * (smp.NVariables + 1) + jCol]);
			res = iFil;
		}
	}
	return res;
 }


__device__ bool enfilarVariablesLibres(TSimplexGPUs &smp, int cnt_columnasFijadas, int &cnt_RestriccionesRedundantes, int &cnt_variablesLiberadas) {

	int jVar, iFil, jCol;

	for (jCol = 0; jCol < smp.NVariables -  cnt_columnasFijadas; jCol++) { // MAP: Antes 1 to nc - 1 - cnt_columnasFijadas, nc - 1 = NVariables
		jVar = -smp.top[jCol];

		if ((jVar > 0) && (smp.flg_x[jVar - 1] == 3)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0
			iFil = buscarMejorPivoteEnCol(smp, jCol, cnt_RestriccionesRedundantes, smp.NRestricciones - 1); // MAP: Modifico indices paso el indice de filaFrom y filaHasta (inclusive)
			if (iFil < 1) {
				return false;
			}
			intercambiar(smp, iFil, jCol);
			cnt_RestriccionesRedundantes++;
			if (iFil > cnt_RestriccionesRedundantes) {
				intercambioFilas(smp, cnt_RestriccionesRedundantes, iFil);
			}
			cnt_variablesLiberadas++;
		}
	}
  
	return true;
}


__device__ int resolverIgualdades(TSimplexGPUs &smp, int &cnt_columnasFijadas, int cnt_varfijas, int &cnt_RestriccionesRedundantes, int cnt_igualdades) {
	int res, cnt_acomodadas, iFila, iColumna, 
		nIgualdadesResueltas, nIgualdadesAResolver, 
		iFilaLibre, iFilaAcomodando;
	int * nCerosFilas;
	int * nCerosCols;
	bool fantasma;
	//string mensajeDeError;

	cnt_acomodadas = cnt_columnasFijadas - cnt_varfijas;

	// Muevo las igualdades que esten en columnas al lado derecho junto con las FIJADAS
	iColumna = smp.NVariables - cnt_columnasFijadas - 1; // MAP: Indice modificado, cambio nc por smp.NVariables 
	while ((iColumna >= 0) && (cnt_acomodadas < cnt_igualdades)) {
		if ((smp.top[iColumna] > 0) && (abs(smp.flg_y[smp.top[iColumna] - 1]) == 2)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0
			if (iColumna != (smp.NVariables - cnt_columnasFijadas - 1)) { // MAP: Indice modificado, cambio nc por smp.NVariables
				intercambioColumnas(smp, iColumna, smp.NVariables - cnt_columnasFijadas - 1); // MAP: Indice modificado, cambio nc por smp.NVariables
			}
			cnt_acomodadas++;
			cnt_columnasFijadas++;
		}
		iColumna--;
	}
	
	// rch@20130307.bugfix - begin ----------------------------
	// ahora reviso las que ya estén declaradas como redundantes a ver si hay
	// igualdades e incremento el contador de acomodadas.
	for (iFilaAcomodando = 0; iFilaAcomodando < cnt_RestriccionesRedundantes; iFilaAcomodando++) { // MAP: originalmente 1 to cnt_RestriccionesRedundantes
		if ((smp.left[iFilaAcomodando] > 0) && (abs(smp.flg_y[iFilaAcomodando]) == 2)) {
			//  Es una restricción  y es de igualdad
			cnt_acomodadas++;
		}
	}

	// En el caso de estar resolviendo un MIPSimplex, en contador de redundantes puede venir
	// incrementado del problema PADRE y pueden haber restricciones de igualdad dentro de las
	// redundantes. Esto hacía que el  "while  cnt_acomodadas < cnt_igualdades"
	// que está unas lineas abajo NO saliera por no alcanzar la condición.
	// rch@20130307.bugfix - end ----------------------------


	// ahora reordeno las igualdades y las que queden en filas las pongo al inicio
	iFilaLibre = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0
	iFilaAcomodando = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0

	while (cnt_acomodadas < cnt_igualdades) {
		if ((smp.left[iFilaAcomodando] > 0) && (abs(smp.flg_y[smp.left[iFilaAcomodando] - 1]) == 2)) { // MAP: Agrego - 1 para mover el indice a la indexacion desde 0
			//vc, dv @20200116 decia flg_y[iFilaAcomodando], se fijaba en cualquier lado
			//  Es una restricción  y es de igualdad
			if (iFilaLibre != iFilaAcomodando) {
				intercambioFilas(smp, iFilaAcomodando, iFilaLibre);
			}
			cnt_acomodadas++;
			iFilaLibre++;
		}
		iFilaAcomodando++;
	} //Al salir de aca iFilaLibre queda en la primer fila que no es de igualdad
	
	res = 1;

	nIgualdadesResueltas = 0;
	nIgualdadesAResolver = iFilaLibre - cnt_RestriccionesRedundantes; // MAP: Antes iFilaLibre - (cnt_RestriccionesRedundantes + 1),  + 1 debido a que iFilaLibre es uno menos que el original
	nCerosFilas = (int*)malloc((smp.NRestricciones + 1)*sizeof(int)); // MAP: antes setLength(nCerosFilas, nf);
	nCerosCols = (int*)malloc((smp.NVariables + 1)*sizeof(int));// MAP: antes setLength(nCerosCols, nc);
	while (nIgualdadesResueltas < nIgualdadesAResolver) {
		//    res:= pasoBuscarFactibleIgualdad( cnt_RestriccionesRedundantes + 1 + nIgualdadesResueltas );
		//    res:= pasoBuscarFactibleIgualdad2( cnt_RestriccionesRedundantes + 1 + nIgualdadesResueltas );
		//    res:= pasoBuscarFactibleIgualdad3( nIgualdadesAResolver - nIgualdadesResueltas);

		// printf("Launch  pasoBuscarFactibleIgualdad4 with:  %d, %d, %d, %d\n", nCerosFilas[0], nCerosCols[0], cnt_columnasFijadas, cnt_RestriccionesRedundantes);
		res = pasoBuscarFactibleIgualdad4(smp, nIgualdadesAResolver - nIgualdadesResueltas, nCerosFilas, nCerosCols, cnt_columnasFijadas, cnt_RestriccionesRedundantes);
		
		if (res ==1) {
			nIgualdadesResueltas = nIgualdadesResueltas + 1;
			cnt_columnasFijadas++;
		} else {
			iFila = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0
			while ((nIgualdadesResueltas < nIgualdadesAResolver) && filaEsFactible(smp, iFila, fantasma)) {
				if (iFila != cnt_RestriccionesRedundantes + 1) {
					intercambioFilas(smp, iFila, cnt_RestriccionesRedundantes + 1);
				}
				cnt_RestriccionesRedundantes++;
				nIgualdadesResueltas++;
				iFila++;
			}
			if (nIgualdadesResueltas < nIgualdadesAResolver) {
				//mensajeDeError = "PROBLEMA INFACTIBLE - Resolviendo igualdades.";
				//printf("%s\n", mensajeDeError.c_str());
				printf("%s\n", "PROBLEMA INFACTIBLE - Resolviendo igualdades.");
				res = -13;
				break;
			} else {
				res = 1;
			}
		}
	}

	free(nCerosFilas); // MAP: Antes setLength(nCerosFilas, 0);
	free(nCerosCols); // MAP: Antes setLength(nCerosCols, 0);
	return res;
}


// Indica si la restricción en kfila esta siendo cumplida
__device__ bool filaEsFactible(TSimplexGPUs &smp, int kfila, bool &fantasma) {
	
	int ix;
	// if e(kfila, nc) < -CasiCero_Simplex then
	if (smp.mat[kfila * (smp.NVariables + 1) + smp.NVariables] < -CasiCero_Simplex) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		// Si la fila es < 0
		fantasma = false;
		return false;
	} else if (smp.left[kfila] < 0) {
		// Si rval es >= 0 reviso si es una variable con cota superior
		ix = -smp.left[kfila];
		// if (flg_x[ix] <> 0) and (e(kfila, nc) > (x_sup.pv[ix] + CasiCero_Simplex_CotaSup)) then
		if ((smp.flg_x[ix - 1] != 0) && ((smp.mat[kfila * (smp.NVariables + 1) + smp.NVariables]) > (smp.x_sup[ix - 1] + CasiCero_Simplex_CotaSup))) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice y agrego -1 para correr el indice
			//Si violo la cota superior
			fantasma = true;
			return false;
		} else {
			fantasma = false;
			return true;
		}
	} else {
		// Si es una y >= 0
		fantasma = false;
		return true;
	}
}


__device__ int pasoBuscarFactibleIgualdad4(TSimplexGPUs &smp, int nIgualdadesNoResueltas, int * nCerosFilas, int * nCerosCols, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes) {

	int iFila, iColumna, columnasLibres,
		filaPiv, colPiv;
	double  maxVal, m;
	
	// Tengo todas las igualdades en columnas al final y las igualdades en filas al principio
	columnasLibres = smp.NVariables - cnt_columnasFijadas; // MAP: remuevo -1 ya que smp.NVariables = nc - 1 
	for (iColumna = 0; iColumna < columnasLibres; iColumna++) { // MAP: Originalmente 1 to columnasLibres
		nCerosCols[iColumna] = 0;
	}
	
	
	// Busco el máximo valor absoluto y cuento la cantidad de ceros en filas y columnas en la caja desde
	// cnt_RestriccionesRedundantes + 1 hasta cnt_RestriccionesRedundantes + nIgualdadesNoResueltas
	// la caja de las igualdades sin resolver
	maxVal = -MaxNReal;
	filaPiv = -1;
	colPiv = -1;
	// MAP: remuevo el +1 debido a la nueva indexacion desde 0, originalmente cnt_RestriccionesRedundantes + 1 to cnt_RestriccionesRedundantes + nIgualdadesNoResueltas
	for (iFila = cnt_RestriccionesRedundantes; iFila < cnt_RestriccionesRedundantes + nIgualdadesNoResueltas; iFila++) { 
		for (iColumna = 0; iColumna <columnasLibres; iColumna++) { // MAP: muevo el indice una lugar hacia atras para considerar el cambio de indexacion desde 0
			m = abs(smp.mat[iFila * (smp.NVariables + 1) + iColumna]);
			if (m < AsumaCero) {
				nCerosFilas[iFila]++;
				nCerosCols[iColumna]++;
			} else if (m > maxVal) {
				maxVal = m;
				filaPiv = iFila;
				colPiv = iColumna;
			}
		}
	}
	
	// Termino de contar la cantidad de ceros en columnas con el resto de las filas
	for (iFila = cnt_RestriccionesRedundantes + nIgualdadesNoResueltas; iFila < smp.NRestricciones; iFila++) { // MAP: cnt_RestriccionesRedundantes + nIgualdadesNoResueltas + 1 to nf - 1
		for (iColumna = 0; iColumna < columnasLibres - 1; iColumna++) { // MAP: Originalmente 1 to columnasLibres - 1
			if (abs(smp.mat[iFila * (smp.NVariables + 1) + iColumna]) < AsumaCero) {
				nCerosCols[iColumna]++;
			}
		}
	}
	
	if (maxVal > CasiCero_Simplex) {
		for (iFila = cnt_RestriccionesRedundantes; iFila < cnt_RestriccionesRedundantes + nIgualdadesNoResueltas; iFila++) { // MAP: Originalmente cnt_RestriccionesRedundantes + 1 to cnt_RestriccionesRedundantes +	nIgualdadesNoResueltas
			for (iColumna = 0; iColumna < columnasLibres; iColumna++) { // MAP: Originalmente 1 to columnasLibres
				if (abs(smp.mat[iFila * (smp.NVariables + 1) + iColumna]) * 10 >= maxVal) {
					// Lo considero como posible pivote
					if ((nCerosFilas[filaPiv] + nCerosCols[colPiv]) < (nCerosFilas[iFila] + nCerosCols[iColumna])) {
						filaPiv = iFila;
						colPiv = iColumna;
					}
				}
			}
		}
		
		// printf("In pasoBuscarFactibleIgualdad4 %d, %d, %d \n", filaPiv, cnt_RestriccionesRedundantes, nIgualdadesNoResueltas);
		// Muevo la fila a intercambiar al final asi me siguen quedando las que voy a acomodar en bloque desde cnt_RestriccionesRedundantes
		if (filaPiv != cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1) { // MAP: Agrego -1 para correr el indice, filaPiv ya esta en el indice correcto
			intercambioFilas(smp, filaPiv, cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1); // MAP: Agrego -1 para correr el indice, filaPiv ya esta en el indice correcto
		}

		intercambiar(smp, cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1, colPiv); // MAP: Agrego -1 para correr el indice, colPiv ya esta en el indice correcto

		if (colPiv != columnasLibres) {
			intercambioColumnas(smp, colPiv, columnasLibres - 1); // MAP: Agrego -1 para correr el indice, colPiv ya esta en el indice correcto
		}
		return 1;
	} else {
		return -1;
	}	
}

__device__ int reordenarPorFactibilidad(TSimplexGPUs &smp, int cnt_RestriccionesRedundantes, int &cnt_RestrInfactibles) {

	int kfil, ix;
	double rval;

	/*
		Primero recorremos las restricciones y
		si la restricción no está violada me fijo si corresponde a una variable
		con restricción de cota superior y si es así verificamos que tampoco esté
		violada la restricción fantasma, si la fantasma se viola hacemos el cambio
		de variable para volverla explícita
	*/
	for (kfil = cnt_RestriccionesRedundantes; kfil < smp.NRestricciones; kfil++) { // MAP: originalmente cnt_RestriccionesRedundantes + 1 to nf - 1
		// rval := e(kfil, nc);
		rval = smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		if (rval > 0) {
		  // Si es = 0 no chequeo pues la fantasma no puede estar violada
			if (smp.left[kfil] < 0) {
				ix = -smp.left[kfil] - 1; // MAP: Agrego -1 para ajustar el indice a la indexacion desde 0
				if ((smp.flg_x[ix] != 0) && (smp.x_sup[ix] < rval)) {
					// Parece que violo la cota superior
					if ((smp.x_sup[ix] + CasiCero_Simplex_CotaSup) < rval) {
						// La viola realmente
						cambiar_borde_de_caja(smp, kfil);
					} else {
						// La viola por errores númericos
						// pon_e(kfil, nc, x_sup.pv[ix])
						smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables] = smp.x_sup[ix]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
					}
				}
			}
		} else {
			if (rval > -CasiCero_Simplex_CotaSup) {
			  // pon_e(kfil, nc, 0);
			  smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables] = 0; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			}
		}
	}

	// Ahora sabemos que las violadas están explícitas, movemos todas las
	//restricciones violadas al final
	kfil = cnt_RestriccionesRedundantes; // MAP: Remuevo + 1 para ajustar el indice a la indexacion desde 0
	cnt_RestrInfactibles = 0;
	while (kfil < (smp.NRestricciones - cnt_RestrInfactibles)) { // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
		// rval:= e(kfil, nc);
		rval = smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		if (rval < 0) {
			cnt_RestrInfactibles++;
			// while (e(nf-cnt_RestrInfactibles, nc ) < 0)
			// MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice, idem para nc	
			while ((smp.mat[(smp.NRestricciones - cnt_RestrInfactibles) * (smp.NVariables + 1) + smp.NVariables] < 0) && (kfil < (smp.NRestricciones - cnt_RestrInfactibles))) {
				cnt_RestrInfactibles++;
			}
			if (kfil < (smp.NRestricciones - cnt_RestrInfactibles)) { // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
				intercambioFilas(smp, kfil, smp.NRestricciones - cnt_RestrInfactibles); // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
			}
		}
		kfil++;
	}
	return cnt_RestrInfactibles;
}

__device__ void cambiar_borde_de_caja(TSimplexGPUs &smp, int k_fila) {
	int ix, k;
	/*
		Realizamos el cambio de variable x'= x_sup - x para que la restricción
		violada sea representada por x' >= 0
		Observar que para la nueva variable la restricción x >= 0 se transforma
		en x' <= x_sup. Es decir que la cota superior de x' es también x_sup.
	*/
	
	// MAP: Confio en el comentario debajo que eso da positivo por lo que resto 1 para ajustar el indice a la indexacion desde 0, de hecho si diera negativo como se trata de un indice fallaria
	ix = -smp.left[k_fila] - 1; // Se supone que esto da positivo, sino no es una x
	for (k = 0; k < smp.NVariables; k++) { // MAP: Muevo los indices 1 to nc-1 un lado a la izquierda para para ajustar el indice a la indexacion desde 0
		smp.mat[k_fila * (smp.NVariables + 1) + k] = -smp.mat[k_fila * (smp.NVariables + 1) + k];
	}
	
	smp.mat[k_fila * (smp.NVariables + 1) + smp.NVariables] = smp.x_sup[ix] - smp.mat[k_fila * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice

	if (abs(smp.flg_x[ix]) != 1) {
		printf("%s\n", "mmmm ... porqué?");
	}
	smp.flg_x[ix] = -smp.flg_x[ix];
  
}


__device__ int pasoBuscarFactible(TSimplexGPUs &smp, int &cnt_RestrInfactibles, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes) {
	
	int pFilaOpt, ppiv, qpiv, ix, res;
	double rval;
	bool filaFantasma, colFantasma;

	pFilaOpt = smp.NRestricciones - cnt_RestrInfactibles; // MAP: Antes nf - cnt_RestrInfactibles, indice ajustado
	// rval:= e(pFilaOpt, nc);
	rval = smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice

	/* OJO LE AGREGO ESTE CHEQUEO PARA PROBAR */
	// Si parece satisfecha verifico que no se esté violándo la  fantasma
	if (rval > 0) {
		if (smp.left[pFilaOpt] < 0) {
			ix = -smp.left[pFilaOpt] - 1; // MAP: Ajusto indice con -1 
			if ((smp.flg_x[ix] != 0) && (rval > smp.x_sup[ix])) {
				if (rval > smp.x_sup[ix] + CasiCero_Simplex) {
					cambiar_borde_de_caja(smp, pFilaOpt);
					// rval:= e(pFilaOpt, nc );
					rval = smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				} else {
					// pon_e(pFilaOpt, nc, x_sup.pv[ix]);
					smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables] = smp.x_sup[ix]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
					rval = smp.x_sup[ix];
				}
			}
		}
	} else if (rval > -CasiCero_Simplex) {
		// pon_e(pFilaOpt, nc, 0);
		smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables] = 0; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		rval = 0;
	}

	if (rval >= 0) {
		// ya es factible, probablemente se arregló con algún cambio anterior.
		cnt_RestrInfactibles--;
		res = 1;
	} else {
		// Nos planteamos el problema de optimización con objetivo el valor de la restricción violada
		if (cnt_RestrInfactibles > 0) {
			qpiv = locate_zpos(smp, pFilaOpt, cnt_columnasFijadas);
			if (qpiv >= 0) { // MAP: Antes > 0, pero con el corrimiento de indice queda >=
				ppiv = mejorpivote(smp, qpiv, pFilaOpt, filaFantasma, colFantasma, true, cnt_RestriccionesRedundantes);
				if (ppiv < 0) { // MAP: Antes < 1, pero con el corrimiento de indice queda < 0
					res = -1; // ShowMessage('No encontre pivote bueno ');
				} else {
					if (!colFantasma) {
						intercambiar(smp, ppiv, qpiv);
						if (filaFantasma) {
							cambio_var_cota_sup_en_columna(smp, qpiv);
						}
						// if ( e( pFilaOpt, nc) >= 0 ) then
						if (smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables] >= 0) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
							cnt_RestrInfactibles--;
						}
						res = 1;
					} else {
						cambio_var_cota_sup_en_columna(smp, ppiv);
						// if ( e( pFilaOpt, nc) >= 0 ) then
						if (smp.mat[pFilaOpt * (smp.NVariables + 1) + smp.NVariables] >= 0) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
							cnt_RestrInfactibles--;
						}
						res = 1;
					}
				}
			} else {
				res = 0; // ShowMessage('No encontre z - positivo ' );
			}
		} else {
			res = -2;
		}

		if (res == -1) {
			// Pruebo si soluciono la infactibildad con un intercambio de la infactible con una de las Activas
			qpiv = locate_qOK(smp, pFilaOpt, smp.NVariables - cnt_columnasFijadas - 1, smp.NVariables, cnt_RestriccionesRedundantes); // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			if (qpiv > 0) {
				intercambiar(smp, pFilaOpt, qpiv);
				cnt_RestrInfactibles--;
				res = 1;
			}
		}
	}

	return res;
}


// Buscamos la columna que en la ultima fila (fila z) tenga el valor positivo mas grande retorna el número de columna si lo encontro, -1 si son todos < 0
// Este paso se da en el Simplex para minimizar, en el de maximizar busca el menos negativo
__device__ int locate_zpos(TSimplexGPUs &smp, int kfila_z, int cnt_columnasFijadas) {
	int j, ires;
	double maxval;
	ires = -1;
	maxval = CasiCero_Simplex;
	for (j = 0; j <= smp.NVariables - cnt_columnasFijadas; j++) { // MAP: Antes 1 to nc - 1 - cnt_columnasFijadas
		if (smp.mat[kfila_z * (smp.NVariables + 1) + j] > maxval) {
			maxval = smp.mat[kfila_z * (smp.NVariables + 1) + j];
			ires = j;
		}
	}
	return ires;
}


// MAP: Este procedimeinto es interno al procedimiento mejorpivote. ESTE PROCEDIMIENTO HABRIA QUE ELIMINARLO Y PEGAR EL CODIGO DIRECTO EN mejorpivote
__device__ void capturarElMejor(double &a_iq_DelMejor, double &a_it_DelMejor, int &p, bool &filaFantasma, bool &colFantasma, double a_iq, double a_it, int i, bool xFantasma_fila) {
  a_iq_DelMejor = a_iq;
  a_it_DelMejor = a_it;
  p = i;
  filaFantasma = xFantasma_fila;
  colFantasma = false; // Solo por si el primero era la Fantasma de la Columna
}


__device__ int mejorpivote(TSimplexGPUs &smp, int q, int kmax, bool &filaFantasma, bool &colFantasma, bool checkearFilaOpt, int cnt_RestriccionesRedundantes) {
  
	int i, p, ix;
	double a_iq, a_it, abs_a_pq,
		a_iq_DelMejor, a_it_DelMejor, abs_a_pq_DelMejor;
	bool xFantasma_fila, esCandidato;

	// inicializaciones no necesarias, solo para evitar el warning
	a_it = 0;
	xFantasma_fila = false;
	a_it_DelMejor = 0;
	a_iq_DelMejor = 1;

	/*11/9/2006 le voy a agregar para que si la q corresponde a una x con manejo
	de cota superior considere la existencia de una fila adicional correspondiente
	a la cota superior.
	Dicha fila tiene un -1 en la coluna q y el valor x_sup como término independiente

	rch.30/3/2007 Agrego el manejo del CasiCero_Simplex

	PA.21/06/2007 Le agrego que al buscar el mejorpivote para optimizar la fila
	kmax chequee si esta es una variable con restriccion de cota superior y que el
	pivote elegido no la viole*/

	ix = -smp.top[q];
	if ((ix > 0) && ( abs(smp.flg_x[ix - 1] ) == 2)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0
		printf("%s\n", "OPATROPA!");
	}

	if ((ix > 0) and ( abs(smp.flg_x[ix - 1] ) == 1 )) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
		// en la columna q hay una x con manejo de cota superior
		colFantasma = true;
		p = q;
		//l o fijamos en -1 porque todas las restricciones fantasma tienen un -1 en
		// en el coeficiente de la variable y x_sup como termino independiente
		a_iq_DelMejor = -1;
		a_it_DelMejor = smp.x_sup[ix - 1]; // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
		abs_a_pq_DelMejor = 1;
	} else {
		p = -1;
		colFantasma = false;
	}

	filaFantasma = false;

	for (i = cnt_RestriccionesRedundantes; i <= kmax - 1; i++) { // MAP: Antes cnt_RestriccionesRedundantes + 1 to kmax - 1, muevo indice para ajustar a la nueva indexacion desde 0, kmax ya viene indexado con el ejuste
		// b(i) >= 0 para todo i / cnt_RestriccionesRedundantes < i < kmax-1
		// Buscamos la fila i que tenga el maximo b(i)/a(i,q) con a(i,q) < 0
		// aiq:= e( i, q );
		a_iq = smp.mat[i * (smp.NVariables + 1) + q];
		if (a_iq >  CasiCero_Simplex) { // Si es positivo, verificamos si se trata de una x y entonces agregamos la fantasma.
			ix = -smp.left[i];
			if ((ix > 0) && (abs(smp.flg_x[ix - 1]) ==1)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
				// La variable en la fila i tiene cota superior, hay que probar con el cambio de variable
				a_iq = -a_iq;
				a_it = smp.x_sup[ix - 1] - smp.mat[i * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				xFantasma_fila = true;
				esCandidato = true;
			} else {
				esCandidato = false;
			}
		} else if (a_iq < -CasiCero_Simplex) { // Si es negativo es candidato
			a_it = smp.mat[i * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			esCandidato = true;
			xFantasma_fila = false;
		} else {
			esCandidato = false;
		}


		if (esCandidato) { // Considero el coeficiente para elegir el pivote
			abs_a_pq = abs(smp.mat[i * (smp.NVariables + 1) + q]); // MAP: Cambio abs( e( i, q ) ); por abs(smp.mat[i][q]));
			/*
			se supone que a_iq < 0 y a_iq_DelMejor < 0 pues sino no son candidatos.
			El término independiente de cualquier fila k, se transformará al usar a_iq como pivote
			como:
			a_k,nc = a_k,nc - a_kq / a_iq * a_i,nc
			y tiene que mantenerce >= 0 para cualquier k <> i.
			a_k,nc - a_kq / a_iq * a_i,nc >= 0  ec.(1)
			dividiendo por a_kq < 0 se tiene
			a_k,nc / a_kq <= a_i,nc / a_iq  ec.(2)
			observar que cada lado de la desigualdad depende solo de los coeficientes
			de la fila k (izquieda) o de la fila i (derecha), Esto nos permite
			ir recorriendo las filas con a_iq < 0 y quedarnos con el de mayor cociente
			a_i,nc/a_iq.

			Para no hacer las divisiones, en lugar de chequear la ec.(2), chequeamos
			la ec.(3) obtenida de la ec.(1) multiplicando por a_iq < 0
			a_k,nc * a_iq <= a_kq  * a_i,nc >= 0  ec.(3)

			Si se cumple la ec.3 a_iq es mejor pivote que a_kp.
			*/
			//aiq < 0 por como lo tomamos para esCandidato
			//bi >= 0 para todo i / cnt_RestriccionesRedundantes < i < kmax-1
			//El pivote es aquel que tenga mayor bi/aiq siempre que bi/aiq < 0 y aiq < 0
			//Ademas bi/aiq y b_max/a_max tienen el mismo signo =>
			//bi/aiq > b_max/a_max <=> bi * a_max > b_max * aiq
			if (p < 0) { // Es el primer candidato
				capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
				
			} else if (( a_it_DelMejor * a_iq) < ( a_it * a_iq_DelMejor ))  {
				capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
				
			} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
				if (abs_a_pq > abs_a_pq_DelMejor) {
					capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
				}
			}
		}
	}
	
	
	//MEJORA de codigo apturarElMejor( i ) es la misma instruccion en todos los casos del if elsif por lo que puede ser englobado en una sola condicion,  i:= kmax; no le veo objeto a esta definicion se puede usar kmax directamente y es mas claro el codigo
	if (checkearFilaOpt) {
		i = kmax;
		ix = -smp.left[kmax];
		if (ix > 0) {  // Es una fila "x"
			if ( abs(smp.flg_x[ix - 1] ) == 1) {   // tiene manejo de cota superior
				// agregamos su fila fantasma como una más candidata a pivotear y a controlar
				// su factibilidad en caso de pivotear con otra.
				// En la fila kmax el aiq es positivo, pues fue elegido con locate_zpos
				//      aiq:= -e(kmax, q);
				a_iq = -smp.mat[kmax * (smp.NVariables + 1) + q];
				a_it = smp.x_sup[ix - 1] - smp.mat[kmax * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				abs_a_pq = abs(a_iq);
				if (a_iq >= 0) printf("%s\n", "aiq >= 0 en tsimplex.mejorpivote");
				//assert(a_iq < 0);
				// static_assert(a_iq < 0, "aiq >= 0 en tsimplex.mejorpivote");
				//      b_:= x_sup.pv[ix] - e(kmax, nc);
				xFantasma_fila = true;
				if (p < 0) { // es el primer candidato
					capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
				} else if (( a_it * a_iq_DelMejor ) > ( a_it_DelMejor * a_iq)) {
					capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
				} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
					if (abs_a_pq > abs_a_pq_DelMejor) {
						capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
					}
				}
			}
		}


		// ahora bien, independientemente de que se trate de una x o una y
		// a( kmax, q ) > 0 por selección del q y a( kmax, ti ) < 0 porque
		// por eso estamos tratando de optimizar chequeando kmax para volverla
		// factible.
		// Agregamos entonces la posibilidad de pivotear con kmax como forma de volverla factible.
		a_iq = smp.mat[kmax * (smp.NVariables + 1) + q];
		a_it = smp.mat[kmax * (smp.NVariables + 1) + smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		xFantasma_fila = false;
		abs_a_pq = abs(a_iq);
		if (p < 0) { // Es el primer candidato
			capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
		} else if (( a_it * a_iq_DelMejor ) < ( a_it_DelMejor * a_iq)) { // OJO, observar que es un "<"
			capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
		} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
			if (abs_a_pq > abs_a_pq_DelMejor) {
				capturarElMejor(a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xFantasma_fila);
			}
		}
	}

	return p;
}


__device__ bool cambio_var_cota_sup_en_columna(TSimplexGPUs &smp, int q) {

	int ix, kfil;
	double xsup;
	bool res;
  
	res = false;
	ix = -smp.top[q];
	// MAP: Agrego -1 para correr el indice a la indexacion desde 0 
	if ((ix > 0) && (abs(smp.flg_x[ix - 1]) == 1)) { // Corresponde a una x con cota sup
		// if abs(flg_x[ix] ) <> 1  then  writeln( 'mmmmm ' );

		// cambio de variable en la misma columna
		smp.flg_x[ix - 1] = -smp.flg_x[ix - 1]; // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
		xsup = smp.x_sup[ix - 1]; // MAP: Agrego -1 para correr el indice a la indexacion desde 0
		
		// MAP: Aca el codigo original esta dividido en 3 casos segun las restricciones redundantes y la variable de compilacion MODIFICAR_REDUNDANTES, como nosotros consideramos la variable siempre activa concateno todo en un caso con un unico for
		for (kfil = 0; kfil <= smp.NRestricciones; kfil++) { // MAP: Antes 1 to cnt_RestriccionesRedundantes
			smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables] = smp.mat[kfil * (smp.NVariables + 1) + smp.NVariables] + smp.mat[kfil * (smp.NVariables + 1) + q] * xsup; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			smp.mat[kfil * (smp.NVariables + 1) + q] = -smp.mat[kfil * (smp.NVariables + 1) + q];
		}

		res = true;
	} else {
		// MAP: IMPLEMENTAR LOS SIGUIENTE QUE ESTA COMENTADO PARA EL MANEJO DE ERROR
		/*
		self.DumpSistemaToXLT_('simplex_quehagoaca.xlt', '' );
		writeln('??? qué hago acá ???');
		raise Exception.Create( 'QUé hago acá' );
		*/
	}
	
	return res;
}


__device__ int locate_qOK(TSimplexGPUs &smp, int p, int jhasta, int jti, int cnt_RestriccionesRedundantes) {
	int mejorq, q;
	double max_apq, apq;

	mejorq = -1;
	max_apq = -1;
	for (q = 0; q <= jhasta; q++) { // MAP: Antes 1 to jhasta, jhasta ya viene con el indice ajustado a la indexacion por 0
		if (test_qOK(smp, p, q, jti, apq, cnt_RestriccionesRedundantes) && ((mejorq < 0) or (apq > max_apq))) {
			mejorq = q;
			max_apq = apq;
		}
	}
	
	return mejorq;
}


/*
Esta función retorna true si la columna q soluciona la infactibilidad
de la fila p. Se supone que (jti) es la columna de los términos constantes
(generalmente la nc ) la dejamos como parámetro por si es necesario

El valor retornado apq, es e(p,q) y puede usarse para
elegir el q que devuelva el valor más grande para disminuir los
errores numéricos.
*/
__device__ bool test_qOK(TSimplexGPUs &smp, int p, int q, int jti, double &apq, int cnt_RestriccionesRedundantes) {
	int k, ix;
	double alfa_p, akq,
		nuevo_ti;

	// apq:= e(p, q);
	apq = smp.mat[p * (smp.NVariables + 1) + q];
	// if ( apq  <= AsumaCero) then
	if (abs(apq) <= AsumaCero) { // rch@202012081043 agrego el abs()
		return false;
	} else {
		// alfa_p:= -e( p, jti ) / apq;
		alfa_p = -smp.mat[p * (smp.NVariables + 1) + jti] / apq;
		ix = -smp.top[q];
		if (ix > 0) { // la col q es una x
			if (smp.flg_x[ix - 1] != 0) {  // tiene manejo de cotasup, MAP: Agrego -1 para correr el indice a la indexacion desde 0 
				if (alfa_p > smp.x_sup[ix - 1]) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
					// De intercambiar esta columna se violaría la cotas superior
					if (alfa_p > (smp.x_sup[ix - 1] + CasiCero_Simplex)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
						return false;
					}
					// else begin //writeln( 'OJO, TSimplex.test_qOk, de intercambiar esta  columna se viola por poquito la cota sup y la dejo pasar' );
				}
			}
		}
		
		for (k = cnt_RestriccionesRedundantes; k < p; k++) { // MAP: cnt_RestriccionesRedundantes + 1 to p - 1, p ya viene con el indice ajustado a la indexacion desde 0
			// akq = e(k, q);
			akq = smp.mat[k * (smp.NVariables + 1) + q];
			// nuevo_ti = e(k,jti) + akq * alfa_p;
			nuevo_ti = smp.mat[k * (smp.NVariables + 1) + jti] + akq * alfa_p;
			if (nuevo_ti < -CasiCero_Simplex) {
				return false;
			} else {
				ix = -smp.left[k];
				if ((ix > 0) and (smp.flg_x[ix - 1] != 0)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
					//  if nuevo_ti > x_sup.pv[ix]  then
					if (nuevo_ti > (smp.x_sup[ix - 1] + CasiCero_Simplex)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
						return false;
					}
				}
			}
		}

		return true;
	}
}


__device__ void darpaso(TSimplexGPUs &smp, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes, int &res) {

	__shared__ int ppiv;
	__shared__ int qpiv;
	__shared__ bool filaFantasma;
	__shared__ bool colFantasma;
	__shared__ bool cambio_var_cota_sup_en_col_result;
	
	// printf("%s\n", "darpaso AAAAAAAAAAAAAAA");

	// cnt_paso++;
	if (threadIdx.x == 0)  qpiv = locate_zpos(smp, smp.NRestricciones, cnt_columnasFijadas); // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
	__syncthreads();
	
	if (qpiv >= 0) { // MAP: Antes > 0, pero con el corrimiento de indice queda >=
		if (threadIdx.x == 0)  ppiv = mejorpivote(smp, qpiv, smp.NRestricciones, filaFantasma, colFantasma, false, cnt_RestriccionesRedundantes); // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
		__syncthreads();
		
		if (ppiv < 0) { // MAP: Antes < 1, pero con el corrimiento de indice queda < 0
			if (threadIdx.x == 0) res = -1;
			return; 
		}
		
		if (!colFantasma) {
			intercambiar_multihilo(smp, ppiv, qpiv);
			__syncthreads();
			
			if (filaFantasma) {
				if (threadIdx.x == 0) {
					cambio_var_cota_sup_en_col_result = cambio_var_cota_sup_en_columna(smp, qpiv);
				}
				__syncthreads();
				
				if (!cambio_var_cota_sup_en_col_result) {
					if (threadIdx.x == 0) res = -1;
					return; 
				}
				
			}
			if (threadIdx.x == 0) res = 1;
		} else {
			if (ppiv != qpiv) printf("%s\n", "Si Es FantasmaDeCol tenía que ser ppiv = qpiv");
			//assert(ppiv == qpiv);
			//static_assert(ppiv = qpiv , "Si Es FantasmaDeCol tenía que ser ppiv = qpiv");
			if (threadIdx.x == 0) {
				cambio_var_cota_sup_en_columna(smp, ppiv);
				res = 1;
			}
		} 
	} else {
		if (threadIdx.x == 0) res = 0; // ShowMessage('No encontre z - positivo ' );
	}

}

