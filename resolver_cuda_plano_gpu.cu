#include <stdio.h>
#include <math.h>
#include "toursimplexmgpus_final.h"

#define BLOCK_SIZE_P_I 64 //  = 2 * WARP SIZE
#define BLOCK_SIZE_GR 160 //  = 5 * WARP SIZE

const double CasiCero_Simplex = 1.0E-7;
// const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

const double M = 1.0E+150; //100; //sqrt(MaxNReal);

const int MAX_VARS = 256; // Esto sera usado para pedir shared memory
const int MAX_RES = 256; // Esto sera usado para pedir shared memory


// 8 * 32 = 256
const int BLOCK_SIZE_E_1X = 32;  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy the arrange of the warp in the block is giving for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y Dx)
const int BLOCK_SIZE_E_1Y = 8;

const int BLOCK_SIZE_E_2X = 1;
const int BLOCK_SIZE_E_2Y = 1;

const int BLOCK_SIZE_E_3X = 1;
const int BLOCK_SIZE_E_3Y = 1;

// 8 * 32 = 128
const int BLOCK_SIZE_E_4X = 32;
const int BLOCK_SIZE_E_4Y = 4;

const int MAX_SIMPLEX_ITERATIONS = 128;


__device__ TSimplexGPUs desestructurarTabloide(TabloideGPUs &tabloide);
__device__ void moverseASolFactible(TabloideGPUs &tabloide);
__device__ void agregarRestriccionesCotaSup(TabloideGPUs &tabloide);
__device__ void agregarVariablesHolguraArtificiales(TabloideGPUs &tabloide);

__device__ void resolver_simplex_big_m(TabloideGPUs &tabloide);

__device__ void locate_min_dj(TSimplexGPUs &smp, int &zpos) ;
__device__ void locate_min_ratio(TSimplexGPUs &smp, int zpos, int &qpos);
__device__ void intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol);

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
	const dim3 DimBlock_e1(BLOCK_SIZE_E_1X, BLOCK_SIZE_E_1Y);
	kernel_resolver_etapa_moverse_a_sol_factible<<< DimGrid_e1, DimBlock_e1, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 1 error", cudaGetErrorString(err));
	
	// const dim3 DimBlock_e2(maxVars, 1);
	const dim3 DimGrid_e2(NTrayectorias, 1);
	const dim3 DimBlock_e2(BLOCK_SIZE_E_2X, BLOCK_SIZE_E_2Y);
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
	const dim3 DimBlock_e3(BLOCK_SIZE_E_3X, BLOCK_SIZE_E_3Y);
	kernel_resolver_etapa_agregar_variables_holgura_artificiales<<< DimGrid_e3, DimBlock_e3, 0, 0 >>>(d_simplex_array);
	cudaDeviceSynchronize();
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("%s: %s\n", "CUDA 3 error", cudaGetErrorString(err));
	
	// cantBloques = ceil((float)NTrayectorias / (float)BLOCK_SIZE_GR);
	// printf("%s: %d\n", "cantBloques enfilar_variables_libres: ", cantBloques);
	// const dim3 DimGrid_e4(cantBloques, 1);
	// const dim3 DimBlock_e4(BLOCK_SIZE_GR, 1);
	const dim3 DimGrid_e4(NTrayectorias, 1);
	const dim3 DimBlock_e4(BLOCK_SIZE_E_4X, BLOCK_SIZE_E_4Y);
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
	/*
		Oprimizacion a desestructurarTabloide
		TSimplexGPUs smp;
		smp.var_x = (int) tabloide[0];
		smp.rest_ini = (int) tabloide[1];
		smp.mat_adv_row = (int) tabloide[2];
		smp.Xb = &tabloide[6*smp.mat_adv_row + 3];
		smp.flg_y = &tabloide[6*smp.mat_adv_row + 1];
		smp.matriz = &tabloide[6*smp.mat_adv_row + 4];
	*/
	
	
	for (int i = threadIdx.y; i < smp.rest_ini; i+= blockDim.y) {
		if (smp.Xb[i*smp.mat_adv_row] < 0) {
			if (threadIdx.y == 0) {
				smp.Xb[i*smp.mat_adv_row] *= -1;
				smp.flg_y[i*smp.mat_adv_row] = (smp.flg_y[i*smp.mat_adv_row] == 0) ? 1 : 2; // Move >= to <=
			}
			for (int j = threadIdx.x; j < smp.var_x; j+= blockDim.x) {
				smp.matriz[i*smp.mat_adv_row + j] *= -1;
			}
		}
	}
	
	//__syncthreads();
}

// No es paralelizable a nivel de bloque. La variable qrest es actualizado en cada bucle, y las celdas a las que se acceden dependen de ese valor, por las actualizaciones a la matriz se deben ejecutar en orden.
__device__ void agregarRestriccionesCotaSup(TabloideGPUs &tabloide) {
	
	TSimplexGPUs smp = desestructurarTabloide(tabloide);
	
	int qrest = smp.rest_ini;
	for (int i = 0; i < smp.var_x; i++) {
		if (smp.flg_x[i] > 0) {
			smp.flg_y[(smp.rest_ini + i)*smp.mat_adv_row] = 1;
			smp.Xb[(smp.rest_ini + i)*smp.mat_adv_row] = smp.sup[i];
			// for (int j = 0; j < smp.var_x; j++) smp.matriz[qrest*smp.mat_adv_row + j] = (qrest == (j + smp.rest_ini))? 1 : 0; los 0's vienen desde la CPU 
			smp.matriz[qrest*smp.mat_adv_row + i] = 1; 
			qrest ++;
		}
	}
	// printf("Rest count %i / %i \n", smp.rest_fin, qrest);
	if (smp.rest_fin != qrest) printf("DISCREPANCIA EN LA CANTIDAD DE RESTRICCIONES FINAL\n");
	
}

// No es paralelizable a nivel de bloque. La variable var_count es actualizado en cada bucle, y las celdas a las que se acceden dependen de ese valor, por las actualizaciones a la matriz se deben ejecutar en orden.
__device__ void agregarVariablesHolguraArtificiales(TabloideGPUs &tabloide) {
	
	TSimplexGPUs smp = desestructurarTabloide(tabloide);
	
	int var_s, var_a, var_count;
	var_s = 0; var_a = 0; var_count = smp.var_x;
	
	for (int i = 0; i < smp.var_x; i++) {
		smp.var_type[i] = 0;
		smp.top[i] = i + 1;
	}
	
	/* Esto ya deberia venir completo desde la CPU
	// Completo con 0s la matriz
	for (int i = 0; i < smp.rest_fin; i++) {
		for (int j = var_count; j < smp.var_all; j++) {
			smp.matriz[i*smp.mat_adv_row + j] = 0;
		}
	}
	*/
	
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
	
	int it;
	__shared__ int zpos, qpos;
	
	if (threadIdx.x == 0 && threadIdx.y == 0) printf("resolver_simplex_big_m_final INT \n");
	
	// printStatus(simplex);
	it = 0;
	
	do {
		locate_min_dj(simplex, zpos);

		if (threadIdx.x == 0 && threadIdx.y == 0) printf("%s %d \n", "zpos", zpos);
		
		if (zpos < 0) {
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				printf("%s %i\n", "Condicion de parada maximo encontrado en iteracion", it);
				printResult(simplex);
			}
			return;
		}
		
		locate_min_ratio(simplex, zpos, qpos);
		
		if (threadIdx.x == 0 && threadIdx.y == 0) printf("%s %d \n", "qpos", qpos);
		
		if (qpos < 0) {
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				printf("%s %i\n", "Posicion de cociente no encontrada en iteracion", it);
				printResult(simplex);
			}
			return;
		}
		
		intercambiarvars(simplex, qpos, zpos);
		__syncthreads();
		
		if (threadIdx.x == 0 && threadIdx.y == 0) printStatus(simplex);
		
		it++;
		
		if (it == MAX_SIMPLEX_ITERATIONS) {
			if (threadIdx.x == 0 && threadIdx.y == 0) printf("Max %i iterations achieved\n", it);
			return;
		}
	} while (true);
	
}


// Cambiar el orden de recorrida y guardar el acumulador en la shared memory, luego usar reduccion para obtener el min
__device__ void locate_min_dj(TSimplexGPUs &smp, int &zpos) {
	__shared__  double apz_acc[MAX_VARS];
	__shared__  int apz_indx[MAX_VARS];
	
	int top;
	int thd_indx = threadIdx.y*blockDim.x + threadIdx.x;

	for (unsigned int z = threadIdx.y; z < smp.var_all; z += blockDim.y) {
		top = smp.top[z] - 1;
		if (smp.var_type[top] != 2) { // it is not an artificial variable
			apz_acc[z] = -smp.z[z];
			apz_indx[z] = z;
			
			// Aca hay que aplicar una reduccion en vez de atomicAdd, e invertir el orden de lectura para que sea coalesced !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
			for (unsigned int y = threadIdx.x; y < smp.rest_fin; y+= blockDim.x) {
				atomicAdd(&apz_acc[z], smp.Cb[y*smp.mat_adv_row] * smp.matriz[y*smp.mat_adv_row + z]);
			}
		} else {
			if (threadIdx.x == 0) apz_indx[z] = -1; // Asi lo excluyo en la reduccion(apz_indx[z] < 0...)
		}
	}
	__syncthreads();
	
	
	// Condicion los hilos en el bloque deben ser mayor o igual que var_all sino hay que agregar un bucle mas para que se proesen el resto del los valores en la reduccion
	if (thd_indx == 0 && smp.var_all > (blockDim.x * blockDim.y)) printf("Condicion los hilos en el bloque deben ser mayor o igual que var_all \n");
	
	// Reduccion
	// Interleaved Addressing
	for (unsigned int s = 1; s < smp.var_all; s *= 2) {
		int index = 2 * s * thd_indx;
		if (index < smp.var_all) {
			if (apz_indx[index + s] >= 0 &&  apz_acc[index + s] < 0 && (apz_indx[index] < 0 || apz_acc[index + s] < apz_acc[index])) {			
				apz_indx[index]  = apz_indx[index + s];
				apz_acc[index] = apz_acc[index + s];
			}
		}
		__syncthreads();
	}
	
	/* Sequential Addressing
	for (unsigned int s = smp.var_all; s > 0; s >>= 1) {
		if (thd_indx < s) {
			if (apz_indx[index + s] >= 0 &&  apz_acc[index + s] && (apz_indx[index] < 0 || apz_acc[index + s] < apz_acc[index])) {			
				apz_indx[index]  = apz_indx[index + s];
				apz_acc[index] = apz_acc[index + s];
			}
		}
		__syncthreads();
	}
	*/
	
	// Escribir resultado
	if (thd_indx == 0) {
		if (apz_indx[0] >= 0 && apz_acc[0] < 0) {
			zpos = apz_indx[0];
		} else {
			zpos = -1;
		}
		// printf("MIn Zj-Cj: %f\n",  apz_acc[0]);
	}
	__syncthreads();
	
}


__device__ void locate_min_ratio(TSimplexGPUs &smp, int zpos, int &qpos) {	
	__shared__  double apy_acc[MAX_RES];
	__shared__  int apy_indx[MAX_RES];
	
	int thd_indx = threadIdx.y*blockDim.x + threadIdx.x;
	
	double denom;
	
	// Condicion los hilos en el bloque deben ser mayor o igual que var_all sino hay que agregar un bucle mas para que se proesen el resto del los valores en la reduccion
	if (thd_indx == 0 && smp.rest_fin > (blockDim.x * blockDim.y)) printf("Condicion los hilos en el bloque deben ser mayor o igual que rest_fin \n");
	
	// Cargo los valores en memoria compartida
	if (thd_indx < smp.rest_fin)  {
		denom = smp.matriz[thd_indx*smp.mat_adv_row + zpos];
		if (denom > CasiCero_Simplex) {
			apy_indx[thd_indx] = thd_indx;
			apy_acc[thd_indx] = smp.Xb[thd_indx*smp.mat_adv_row] / denom;
		} else {	
			apy_indx[thd_indx] = -1; // Asi lo excluyo en la reduccion(apy_indx[z] < 0...)
		}
	}
	
	__syncthreads();
	
	// Reduccion
	// Interleaved Addressing
	for (unsigned int s = 1; s < smp.rest_fin; s *= 2) {
		int index = 2 * s * thd_indx;
		if (index < smp.rest_fin) {			
			if (apy_indx[index + s] >= 0 &&  apy_acc[index + s]  > -CasiCero_Simplex && (apy_indx[index] < 0 || apy_acc[index + s] < apy_acc[index])) {			
				apy_indx[index]  = apy_indx[index + s];
				apy_acc[index] = apy_acc[index + s];
			}
		}
		__syncthreads();
	}
	
	// Escribir resultado
	if (thd_indx == 0) {
		if (apy_indx[0] >= 0 && apy_acc[0]  > -CasiCero_Simplex) {
			qpos = apy_indx[0];
		} else {
			qpos = -1;
		}
		// printf("Min Q: %f\n",  apy_acc[0]);
	}
	__syncthreads();

}

__device__ void intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol) {

	double m, invPiv;
	int i, j, ipos, k;
	
	invPiv = 1 / smp.matriz[kfil * smp.mat_adv_row + jcol];
	
	int thd_indx = threadIdx.y*blockDim.x + threadIdx.x;
	int block_dim = blockDim.x * blockDim.y;
	
	ipos = kfil * smp.mat_adv_row;
	smp.Xb[kfil*smp.mat_adv_row] *= invPiv; // Modifico Xb
	for (j = thd_indx; j < smp.var_all; j+= block_dim) { // Modifico la fila k
		smp.matriz[ipos + j] *= invPiv;
	}
	
	for (i = thd_indx; j < smp.rest_fin; i+= block_dim) { // Modifico la columna j
		smp.matriz[i*smp.mat_adv_row + jcol] = 0; 
	}
	
	if (thd_indx == 0) smp.matriz[kfil * smp.mat_adv_row + jcol] = 1; // Modifico el pivote
	
	for (i = threadIdx.y; i < smp.rest_fin; i += blockDim.y) {
		if (i != kfil) {
			m = smp.matriz[i *smp.mat_adv_row + jcol];
			if (threadIdx.x == 0) {
				smp.Xb[i*smp.mat_adv_row] -= m*smp.Xb[kfil*smp.mat_adv_row]; // Modifico Xb
			}
			
			for (j = threadIdx.x; j < smp.var_all; j += blockDim.x) { // Modifico la Matriz
				if (j != jcol) {
					smp.matriz[i *smp.mat_adv_row + j] -= m * smp.matriz[kfil*smp.mat_adv_row + j]; // Aca esta actualizacion se hace coalesced y como es la mas importante podemos decir que el acceso es coalesced mayoritariamente
				} // else {
					// smp.matriz[i*smp.mat_adv_row + j] = 0; // Esto lo saque para afuera para generar menos divergencia
				// }
			}
		}
	}
	
	if (thd_indx == 0) {
		k = smp.top[jcol];
		smp.top[jcol] = smp.left[kfil*smp.mat_adv_row];
		smp.left[kfil*smp.mat_adv_row] = k;
		
		smp.Cb[kfil*smp.mat_adv_row] = smp.z[jcol];
	}
	
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
