#include <stdio.h>
#include <math.h>
#include "toursimplexmgpus_final.h"

#define BLOCK_SIZE_P_I 64 //  = 2 * WARP SIZE
#define BLOCK_SIZE_GR 160 //  = 5 * WARP SIZE

const double CasiCero_Simplex = 1.0E-7;
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

const double M = 1.0E+150; //100; //sqrt(MaxNReal);

__device__ TSimplexGPUs desestructurarTabloide(TabloideGPUs &tabloide);
__device__ void moverseASolFactible(TabloideGPUs &tabloide);
__device__ void agregarRestriccionesCotaSup(TabloideGPUs &tabloide);
__device__ void agregarVariablesHolguraArtificiales(TabloideGPUs &tabloide);

__device__ void resolver_simplex_big_m(TabloideGPUs &tabloide);

__device__ int locate_min_dj(TSimplexGPUs &smp);
__device__ int locate_min_ratio(TSimplexGPUs &smp, int zpos);
__device__ bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol);

__device__ void printStatus(TSimplexGPUs &smp);
__device__ void printResult(TSimplexGPUs &smp);
__device__ double findVarXbValue(TSimplexGPUs &smp, int indx);
__device__ int findVarIndex(TSimplexGPUs &smp, int indx);

__global__ void kernel_resolver_etapa_moverse_a_sol_factible(TDAOfSimplexGPUs simplex_array) {
	moverseASolFactible(simplex_array[blockIdx.x]);
} 

__global__ void kernel_resolver_etapa_agregar_restricciones_cota_sup(TDAOfSimplexGPUs simplex_array) {
	agregarRestriccionesCotaSup(simplex_array[blockIdx.x]);
} 

__global__ void kernel_resolver_etapa_agregar_variables_holgura_artificiales(TDAOfSimplexGPUs simplex_array) {
	agregarVariablesHolguraArtificiales(simplex_array[blockIdx.x]);
}

__global__ void kernel_resolver_simplex_big_m(TDAOfSimplexGPUs simplex_array) {
	resolver_simplex_big_m(simplex_array[blockIdx.x]);
}


extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	int largo, alto, n_vars, n_rest;
	
	int maxVars = 0;
	int maxRest = 0;
	
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		largo = (int) simplex_array[kTrayectoria][3];
		n_vars = (int) simplex_array[kTrayectoria][largo];
		n_rest = (int) simplex_array[kTrayectoria][largo + 1];
		alto = n_rest + 6;
		
		if (n_vars > maxVars) maxVars = n_vars;
		if (n_rest > maxRest) maxRest = n_rest;
		
		cudaMemcpy(h_simplex_array[kTrayectoria], simplex_array[kTrayectoria], largo*alto*sizeof(double), cudaMemcpyHostToDevice);
		
	}

	cudaMemcpy(d_simplex_array, h_simplex_array, NTrayectorias*sizeof(TabloideGPUs), cudaMemcpyHostToDevice);
	
	cudaError_t err;
	
	// Ejecuto los kernels
	// const dim3 DimBlock_e1(maxRest, 1);
	const dim3 DimGrid_e1(NTrayectorias, 1);
	const dim3 DimBlock_e1(1, 1);
	kernel_resolver_etapa_moverse_a_sol_factible<<< DimGrid_e1, DimBlock_e1, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 1 error", cudaGetErrorString(err));
	
	// const dim3 DimBlock_e2(maxVars, 1);
	const dim3 DimGrid_e2(NTrayectorias, 1);
	const dim3 DimBlock_e2(1, 1);
	kernel_resolver_etapa_agregar_restricciones_cota_sup<<< DimGrid_e2, DimBlock_e2, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 2 error", cudaGetErrorString(err));
	
	// int cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	// printf("%s: %d\n", "cantBloques fijar_variables: ", cantBloques);
	// const dim3 DimGrid_e3(cantBloques, 1);
	// const dim3 DimGrid_e3(1, 1);
	// const dim3 DimBlock_e3(BLOCK_SIZE_GR, 1);
	const dim3 DimGrid_e3(NTrayectorias, 1);
	const dim3 DimBlock_e3(1, 1);
	kernel_resolver_etapa_agregar_variables_holgura_artificiales<<< DimGrid_e3, DimBlock_e3, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 3 error", cudaGetErrorString(err));
	
	// cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	// printf("%s: %d\n", "cantBloques enfilar_variables_libres: ", cantBloques);
	// const dim3 DimGrid_e4(cantBloques, 1);
	// const dim3 DimBlock_e4(BLOCK_SIZE_GR, 1);
	const dim3 DimGrid_e4(NTrayectorias, 1);
	const dim3 DimBlock_e4(1, 1);
	kernel_resolver_simplex_big_m<<< DimGrid_e4, DimBlock_e4, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 4 error", cudaGetErrorString(err));
	
	cudaMemcpy(h_simplex_array, d_simplex_array, NTrayectorias*sizeof(TabloideGPUs), cudaMemcpyDeviceToHost);

	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		largo = (int) simplex_array[kTrayectoria][3];
		alto = (int) simplex_array[kTrayectoria][largo + 1] + 6;
		cudaMemcpy(simplex_array[kTrayectoria], h_simplex_array[kTrayectoria], largo*alto*sizeof(double), cudaMemcpyDeviceToHost);
	}

}

/**************************************************************************************************************************************************************************************************/

__device__ TSimplexGPUs desestructurarTabloide(TabloideGPUs &tabloide) {
	TSimplexGPUs smp;
	
	smp.tabloide = tabloide;
	
	smp.var_x = (int) tabloide[0];
	smp.rest_ini = (int) tabloide[1];
	smp.mat_adv_row = (int) tabloide[2];
	smp.var_all = (int) tabloide[smp.mat_adv_row];
	smp.rest_fin = (int) tabloide[smp.mat_adv_row + 1];
	
	smp.z = &tabloide[4]; // funcion z, cantidad de variables, horizontal
    smp.flg_x = &tabloide[smp.mat_adv_row + 4]; // cantidad de variables x = smp.var_x
	
	smp.sup = &tabloide[2*smp.mat_adv_row + 4]; // cantidad de variables x = smp.var_x, horizontal
	smp.inf = &tabloide[3*smp.mat_adv_row + 4]; // cantidad de variables x = smp.var_x, horizontal
	
	smp.var_type = &tabloide[4*smp.mat_adv_row + 4]; // largo filas, horizontal
	
	smp.top = &tabloide[5*smp.mat_adv_row + 4]; // largo filas, horizontal
    smp.left = &tabloide[6*smp.mat_adv_row]; // largo restricciones finales, vertical
	
	smp.flg_y = &tabloide[6*smp.mat_adv_row + 1]; // 0 restriccion >=, 1 <=, 2 =, vertical
	
	
	smp.Cb = &tabloide[6*smp.mat_adv_row + 2]; // cantidad de restricciones, vertical
	smp.Xb = &tabloide[6*smp.mat_adv_row + 3]; // cantidad de restricciones, vertical
	
    smp.matriz = &tabloide[6*smp.mat_adv_row + 4]; 
	
	return smp;
}

__device__ void moverseASolFactible(TabloideGPUs &tabloide) {
	
	TSimplexGPUs smp = desestructurarTabloide(tabloide);
	
	for (int i = 0; i < smp.rest_ini; i++) {
		if (smp.Xb[i*smp.mat_adv_row] < 0) {
			smp.Xb[i*smp.mat_adv_row] *= -1;
			for (int j = 0; j < smp.var_x; j++) {
				smp.matriz[i*smp.mat_adv_row + j] *= -1;
			}
			smp.flg_y[i*smp.mat_adv_row] = (smp.flg_y[i*smp.mat_adv_row] == 0) ? 1 : 2; // Move >= to <=
		}
	}
	
	//__syncthreads();
}

__device__ void agregarRestriccionesCotaSup(TabloideGPUs &tabloide) {
	
	TSimplexGPUs smp = desestructurarTabloide(tabloide);
	
	int qrest = smp.rest_ini;
	for (int i = 0; i < smp.var_x; i++) {
		if (smp.flg_x[i] > 0) {
			smp.flg_y[(smp.rest_ini + i)*smp.mat_adv_row] = 1;
			smp.Xb[(smp.rest_ini + i)*smp.mat_adv_row] = smp.sup[i];
			for (int j = 0; j < smp.var_x; j++) smp.matriz[qrest*smp.mat_adv_row + j] = (qrest == (j + smp.rest_ini))? 1 : 0;
			qrest ++;
		}
	}
	// printf("Rest count %i / %i \n", smp.rest_fin, qrest);
	if (smp.rest_fin != qrest) printf("DISCREPANCIA EN LA CANTIDAD DE RESTRICCIONES FINAL\n");
	
}

__device__ void agregarVariablesHolguraArtificiales(TabloideGPUs &tabloide) {
	
	TSimplexGPUs smp = desestructurarTabloide(tabloide);
	
	int var_s, var_a, var_count;
	var_s = 0; var_a = 0; var_count = smp.var_x;
	
	for (int i = 0; i < smp.var_x; i++) {
		smp.var_type[i] = 0;
		smp.top[i] = i + 1;
	}
	
	// Completo con 0s la matriz
	for (int i = 0; i < smp.rest_fin; i++) {
		for (int j = var_count; j < smp.var_all; j++) {
			smp.matriz[i*smp.mat_adv_row + j] = 0;
		}
	}
	
	for (int i = 0; i < smp.rest_fin; i++) {
		if (smp.flg_y[i*smp.mat_adv_row] == 0) { // rest >=
			smp.matriz[i*smp.mat_adv_row + var_count] = -1;
			smp.matriz[i*smp.mat_adv_row + var_count +1] = 1;
			
			smp.var_type[var_count] = 1;
			smp.var_type[var_count +1] = 2;
			
			smp.z[var_count] = 0;
			smp.z[var_count +1] = -M;
			
			smp.top[var_count] = var_count + 1;
			smp.top[var_count + 1] = var_count + 2;
			smp.left[i*smp.mat_adv_row] = var_count + 2;
			
			smp.Cb[i*smp.mat_adv_row] = -M;
			
			var_s++; var_a++; var_count += 2;
		} else if (smp.flg_y[i*smp.mat_adv_row] == 1) { // rest <=
			smp.matriz[i*smp.mat_adv_row + var_count] = 1;
			
			smp.var_type[var_count] = 1;
			
			smp.z[var_count] = 0;
			
			smp.top[var_count] = var_count + 1;
			smp.left[i*smp.mat_adv_row] = var_count + 1;
			
			smp.Cb[i*smp.mat_adv_row] = 0;
			
			var_a++; var_count ++;
		} else { // 2: rest =
			smp.matriz[i*smp.mat_adv_row + var_count] = 1;
			
			smp.var_type[var_count] = 2;
			
			smp.z[var_count] = -M;
			
			smp.top[var_count] = var_count + 1;
			smp.left[i*smp.mat_adv_row] = var_count + 1;
			
			smp.Cb[i*smp.mat_adv_row] = -M;
			
			var_s++; var_count ++;
		}
	}
	
	//printf("Var count %i / %i \n", smp.var_all, var_count);
	if (smp.var_all != var_count) printf("DISCREPANCIA EN LA CANTIDAD DE VARIABLES FINAL\n");
	
}

__device__ void resolver_simplex_big_m(TabloideGPUs &tabloide) {
	
	TSimplexGPUs simplex = desestructurarTabloide(tabloide);
	
	int zpos, qpos, it;
	
	printf("resolver_simplex_big_m_final INT \n");
	
	printStatus(simplex);
	it = 0;
	
	do {
		zpos = locate_min_dj(simplex);
		printf("%s %d \n", "zpos", zpos);
		
		if (zpos < 0) {
			printf("%s %i\n", "Condicion de parada maximo encontrado en iteracion", it);
			printResult(simplex);
			return;
		}
		
		qpos = locate_min_ratio(simplex, zpos);
		printf("%s %d \n", "qpos", qpos);
		if (qpos < 0) {
			printf("%s %i\n", "Posicion de cociente no encontrada en iteracion", it);
			return;
		}
		
		intercambiarvars(simplex, qpos, zpos);
		
		printStatus(simplex);
		
		it++;
		
		if (it == 128) {
			printf("Max %i iterations achieved\n", it);
			return;
		}
	} while (true);
	
}

__device__ int locate_min_dj(TSimplexGPUs &smp) {
	int mejorz, z, y, top;
	double min_apz, apz;

	mejorz = -1;
	min_apz = 0;
	for (z = 0; z < smp.var_all; z++) {
		top = smp.top[z] - 1;
		if (smp.var_type[top] != 2) { // it is not an artificial variable
			apz = -smp.z[z];
			for (y = 0; y < smp.rest_fin; y++) {
				apz += smp.Cb[y*smp.mat_adv_row] * smp.matriz[y*smp.mat_adv_row + z]; // Cj
			}
			if (apz < 0 && apz < min_apz) {
				mejorz = z;
				min_apz = apz;
				// printf("MIn Zj-Cj: %f\n",  min_apz);
			}
		}
	}
	
	return mejorz;
}


__device__ int locate_min_ratio(TSimplexGPUs &smp, int zpos) {
	int mejory, y;
	double min_apy, qy, denom;

	mejory = -1;
	min_apy = MaxNReal;
	printf("qy:\t");
	for (y = 0; y < smp.rest_fin; y++) {
		denom = smp.matriz[y*smp.mat_adv_row + zpos];
		// printf("%.1f / %.1f ",  smp.Xb[y*smp.mat_adv_row], denom);
		// printf("Denominador: %f\n",  denom);
		if (denom > CasiCero_Simplex) {
			qy = smp.Xb[y*smp.mat_adv_row] / denom;
			printf(" (%.1f)\t",  qy);
			if (qy > -CasiCero_Simplex && qy < min_apy) {
				mejory = y;
				min_apy = qy;
			}
		} else {
			printf(" (NA)\t");
		}
	}
	// printf("Min Q: %f\n",  min_apy);
	return mejory;
	
}

__device__ bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol) {

	double m, invPiv;
	int i, j, ipos, k;
	
	invPiv = 1 / smp.matriz[kfil * smp.mat_adv_row + jcol];
	
	ipos = kfil * smp.mat_adv_row;
	smp.Xb[kfil*smp.mat_adv_row] *= invPiv; // Modifico Xb
	for (j = 0; j < smp.var_all; j++) { // Modifico la k fila
		smp.matriz[ipos + j] *= invPiv;
	}
	smp.matriz[kfil * smp.mat_adv_row + jcol] = 1;
	
	for (i = 0; i < smp.rest_fin; i++) {
		if (i != kfil) {
			m = smp.matriz[i *smp.mat_adv_row + jcol];
			
			smp.Xb[i*smp.mat_adv_row] -= m*smp.Xb[kfil*smp.mat_adv_row]; // Modifico Xb
			for (j = 0; j < smp.var_all; j++) { // Modifico la Matriz
				if (j != jcol) {
					smp.matriz[i *smp.mat_adv_row + j] -= m * smp.matriz[kfil*smp.mat_adv_row + j]; 
				} else {
					smp.matriz[i*smp.mat_adv_row + j] = 0;
				}
			}
		}
	}
	
	k = smp.top[jcol];
	smp.top[jcol] = smp.left[kfil*smp.mat_adv_row];
	smp.left[kfil*smp.mat_adv_row] = k;
	
	smp.Cb[kfil*smp.mat_adv_row] = smp.z[jcol];
	
	return true;
 }
 
__device__ void printStatus(TSimplexGPUs &smp) {
	printf("%s, (%i, %i)\n", "Tabloide", smp.rest_fin + 6, smp.mat_adv_row);
	for(int i = 0; i < smp.rest_fin + 6; i++) {
		for(int j = 0; j < smp.mat_adv_row; j++) {
			printf("%.2f\t", (double) smp.tabloide[i*smp.mat_adv_row + j] );
			//printf("%E \t", smp.tabloide[i*smp.NColumnas + j] );
			//printf("(%i,%i,%i)%f  \t", i, j, (i*smp.NColumnas) + j, smp.tabloide[(i*smp.NColumnas) + j]);
		}
		printf("\n");
	}
	
}

__device__ void printResult(TSimplexGPUs &smp) {
	printf("%s\n", "Resultado");
	double bi, val;
	double min = 0;
	int varType;
	
	for(int i = 0; i < smp.var_x; i++) {
		bi = findVarXbValue(smp, i);
		val = bi + smp.inf[i];
		if (val != 0 || bi != 0) {
			printf("x%i = %.12f  (Xbi = %.12f)\n", findVarIndex(smp, i),  val, bi);
			min -= val * smp.z[i];
		}
	}
	
	for(int i = smp.var_x; i < smp.var_all; i++) {
		bi = findVarXbValue(smp, i);
		if (bi != 0) {
			varType = smp.var_type[i];
			if (varType == 1) {
				val = bi;
				printf("s%i = %.12f \n", findVarIndex(smp, i), val);
			} else {
				val = bi;
				printf("a - error%i = %.12f\n", findVarIndex(smp, i),  val);
			}
		}
	}
	
	printf("Z min = %.2f \n", -min);
	
}

__device__ double findVarXbValue(TSimplexGPUs &smp, int indx) {
	int lefti;
	for(int i = 0; i < smp.rest_fin; i++) {
		lefti = ((int) smp.left[i*smp.mat_adv_row]);
		if (indx == (lefti - 1)) {
			return smp.Xb[i*smp.mat_adv_row];
		}
	}
	return 0;
}

__device__ int findVarIndex(TSimplexGPUs &smp, int indx) {
	int vind = 1;
	int varType = smp.var_type[indx];
	if ( varType == 0) {
		return indx  + 1;
	} else {
		for (int i = 0; i < indx; i++) {
			if (smp.var_type[i] == varType) vind++;
		}
	}
	return vind;
}
