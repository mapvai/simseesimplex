#include <stdio.h>
#include <math.h>
#include <stdlib.h>		/* abs */
#include <assert.h>	/* assert */
#include <iostream>
#include <string>
using namespace std;

#include "toursimplexmgpus_final.h"

const double CasiCero_Simplex = 1.0E-7;
// const double AsumaCero =  1.0E-16; // EPSILON de la maquina en cuentas con Double, CONFIRMAR SI ESTO ES CORRECTO (double 64 bits, 11 exponente y 53 mantisa, 53 log10(2) ≈ 15.955 => 2E−53 ≈ 1.11 × 10E−16 => EPSILON  ≈ 1.0E-16)
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

const double M = 100; //100; //sqrt(MaxNReal);

void resolver_cpu(TabloideGPUs &simplex) ;
TSimplexGPUs desestructurarTabloide(TabloideGPUs &tabloide);
void moverseASolFactible(TSimplexGPUs &smp);
void agregarRestriccionesCotaSup(TSimplexGPUs &smp);
void agregarVariablesHolguraArtificiales(TSimplexGPUs &smp);
void resolver_simplex_big_m(TSimplexGPUs &simplex) ;
bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol);
int locate_min_dj(TSimplexGPUs &smp);
int locate_min_ratio(TSimplexGPUs &smp, int zpos);
void resolver_ejemplo1();
void resolver_ejemplo2trasnform();
void printStatus(TSimplexGPUs &smp);
void printResult(TSimplexGPUs &smp);
double findVarXbValue(TSimplexGPUs &smp, int indx);
int findVarIndex(TSimplexGPUs &smp, int indx);

extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		resolver_cpu(simplex_array[kTrayectoria]);
	}
	
	// resolver_ejemplo2trasnform();
	
}

void resolver_ejemplo1() {
	
/*
	Problema Propuesto por SimSEE
		Min x1 + 3x2 + 2x3
		st:
		x1 + x2 + x3 	≥ -10.5
		x1 + x2 			= - 5.3
		x1  		- x3 		≤ 2.9

		0 ≤ x1 ≤ 12,  -6 ≤ x2 ≤ 6, -5 ≤ x3 ≤ 5
		
		Sol SIMSEE: x1 = 0, x2 = -5.3, x3 = -2.9 Verificado, z min = -21.7
		
=> cambio variable para las cotas inferiores xc = x + cota inf => x = xc - cota inf => Sol xc: x1 = 0, x2 = 0.7, x3 = 2.1
	Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	≥ -10.5 + 6 + 5 = 0.5
		x1 + x2 			= - 5.3 + 6 		 = 0.7
		x1  		  - x3 	≤ 2.9 - 5 			 = -2.1
		x1					≤ 12
				x2			≤ 6 + 6 			 = 12
						x3	≤ 5 + 5				 = 10
						
		x1, x2, x3 > 0
		
=>	Move to a factible solution (Xb > 0)
		Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	≥ 0.5
		x1 + x2 			= 0.7
	   -x1  		 + x3 	≥ 2.1
		x1					≤ 12
				x2			≤ 12
						x3	≤ 10
						
		x1, x2, x3 > 0
	
=> Agregamos las variables de holgura y demasia 
		Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	 - s1 + a1  = 0.5
		x1 + x2 			+ a2          = 0.7
	 - x1  		 + x3 	 - s2 + a3 = 2.1
		x1					+ s3 		 = 12
				x2			+ s4 		 = 12
						x3	+ s5 		 = 10
	
	x1..s5 ≥ 0
	
	RESULTADO OUR SIMPLEX: x1 = 0.7, xc2 = 0 => x2 = 0 - 6 = -6, xc3 = 2.8 => x3 = 2.8 - 5 = -2.2 Verificado, da tambien z min = -21.7
*/	
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
		3, 3, 15, 0, -1, -3, -2, 0, -M, -M, 0, -M, 0, 0, 0, 
		11, 6, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 12, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, -6, -5, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 
		0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
		5, 0, -M, 0.5, 1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 
		6, 2, -M, 0.7, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
		8, 0, -M, 2.1, -1, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 
		9, 1, 0, 12, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
		10, 1, 0, 12, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
		11, 1, 0, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
	};
	
	TabloideGPUs tabloide = (double*)&tabl;
	TSimplexGPUs simplex = desestructurarTabloide(tabloide);
	resolver_simplex_big_m(simplex);
}

void resolver_ejemplo2trasnform() {
	
/*
	Problema Propuesto por SimSEE
		Min x1 + 3x2 + 2x3
		st:
		x1 + x2 + x3 	≥ -10.5
		x1 + x2 			= - 5.3
		x1  		- x3 		≤ 2.9

		0 ≤ x1 ≤ 12,  -6 ≤ x2 ≤ 6, -5 ≤ x3 ≤ 5
		
		Sol SIMSEE: x1 = 0, x2 = -5.3, x3 = -2.9 Verificado, z min = -21.7
		
=> cambio variable para las cotas inferiores xc = x + cota inf => x = xc - cota inf => Sol xc: x1 = 0, x2 = 0.7, x3 = 2.1
	Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	≥ -10.5 + 6 + 5 = 0.5
		x1 + x2 			= - 5.3 + 6 		 = 0.7
		x1  		  - x3 	≤ 2.9 - 5 			 = -2.1
		x1					≤ 12
				x2			≤ 6 + 6 			 = 12
						x3	≤ 5 + 5				 = 10
						
		x1, x2, x3 > 0
*/	
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
		3, 3, 15, 0, -1, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 
		11, 6, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 12, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, -6, -5, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0.5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 2, 0, 0.7, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 2.1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};
	
	TabloideGPUs tabloide = (double*)&tabl;
	TSimplexGPUs simplex = desestructurarTabloide(tabloide);
	
	/*
	=>	Move to a factible solution (Xb > 0)
		Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	≥ 0.5
		x1 + x2 			= 0.7
	   -x1  		 + x3 	≥ 2.1
		x1					≤ 12
				x2			≤ 12
						x3	≤ 10
						
		x1, x2, x3 > 0
	
=> Agregamos las variables de holgura y demasia 
		Max -x1 - 3x2 - 2x3
		st:
		x1 + x2 + x3 	 - s1 + a1  = 0.5
		x1 + x2 			+ a2          = 0.7
	 - x1  		 + x3 	 - s2 + a3 = 2.1
		x1					+ s3 		 = 12
				x2			+ s4 		 = 12
						x3	+ s5 		 = 10
	
	x1..s5 ≥ 0
	
	RESULTADO OUR SIMPLEX: x1 = 0.7, xc2 = 0 => x2 = 0 - 6 = -6, xc3 = 2.8 => x3 = 2.8 - 5 = -2.2 Verificado, da tambien z min = -21.7
	*/
	
	moverseASolFactible(simplex);
	
	agregarRestriccionesCotaSup(simplex);
	
	agregarVariablesHolguraArtificiales(simplex);
	
	resolver_simplex_big_m(simplex);

}

void resolver_cpu(TabloideGPUs &tabloide) {
	TSimplexGPUs simplex = desestructurarTabloide(tabloide);
	
	moverseASolFactible(simplex);
	agregarRestriccionesCotaSup(simplex);
	agregarVariablesHolguraArtificiales(simplex);
	
	resolver_simplex_big_m(simplex);
}

TSimplexGPUs desestructurarTabloide(TabloideGPUs &tabloide) {
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

void moverseASolFactible(TSimplexGPUs &smp) {
	for (int i = 0; i < smp.rest_ini; i++) {
		if (smp.Xb[i*smp.mat_adv_row] < 0) {
			smp.Xb[i*smp.mat_adv_row] *= -1;
			for (int j = 0; j < smp.var_x; j++) {
				smp.matriz[i*smp.mat_adv_row + j] *= -1;
			}
			smp.flg_y[i*smp.mat_adv_row] = (smp.flg_y[i*smp.mat_adv_row] == 0) ? 1 : 2; // Move >= to <=
		}
	}
}

void agregarRestriccionesCotaSup(TSimplexGPUs &smp) {
	int qrest = smp.rest_ini;
	for (int i = 0; i < smp.var_x; i++) {
		if (smp.flg_x[i] == 1) {
			smp.flg_y[(smp.rest_ini + i)*smp.mat_adv_row] = 1;
			smp.Xb[(smp.rest_ini + i)*smp.mat_adv_row] = smp.sup[i];
			for (int j = 0; j < smp.var_x; j++) smp.matriz[qrest*smp.mat_adv_row + j] = (qrest == (j + smp.rest_ini))? 1 : 0;
			qrest ++;
		}
	}
	printf("%i / %i \n", smp.rest_fin, smp.rest_ini + qrest);
	if (smp.rest_fin != qrest) printf("DISCREPANCIA EN LA CANTIDAD DE RESTRICCIONES FINAL\n");
}

void agregarVariablesHolguraArtificiales(TSimplexGPUs &smp) {
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
	
	if (smp.var_all != var_count) printf("DISCREPANCIA EN LA CANTIDAD DE VARIABLES FINAL\n");
	
}

void resolver_simplex_big_m(TSimplexGPUs &simplex) {
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
		
		if (it == 14) {
			printf("Max %i iterations achieved\n", it);
			return;
		}
	} while (true);
	
}

int locate_min_dj(TSimplexGPUs &smp) {
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
				printf("MIn Zj-Cj: %f\n",  min_apz);
			}
		}
	}
	
	return mejorz;
}


int locate_min_ratio(TSimplexGPUs &smp, int zpos) {
	int mejory, y;
	double min_apy, qy, denom;

	mejory = -1;
	min_apy = MaxNReal;
	printf("qy:\t");
	for (y = 0; y < smp.rest_fin; y++) {
		denom = smp.matriz[y*smp.mat_adv_row + zpos];
		printf("%.1f / %.1f ",  smp.Xb[y*smp.mat_adv_row], denom);
		// printf("Denominador: %f\n",  denom);
		if (denom > CasiCero_Simplex) {
			qy = smp.Xb[y*smp.mat_adv_row] / denom;
			printf(" (%.1f)\t",  qy);
			if (qy > 0 && qy < min_apy) {
				mejory = y;
				min_apy = qy;
			}
		} else {
			printf(" (NA)\t");
		}
	}
	printf("Min Q: %f\n",  min_apy);
	return mejory;
	
}

bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol) {

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
	
	smp.Cb[kfil*smp.mat_adv_row] = smp.z[jcol] ;
	
	return true;
 }


void printStatus(TSimplexGPUs &smp) {
	printf("%s, (%i, %i)\n", "Tabloide", smp.rest_fin + 6, smp.mat_adv_row);
	for(int i = 0; i < smp.rest_fin + 6; i++) {
		for(int j = 0; j < smp.mat_adv_row; j++) {
			printf("%.1f\t", smp.tabloide[i*smp.mat_adv_row + j] );
			//printf("%E \t", smp.tabloide[i*smp.NColumnas + j] );
			//printf("(%i,%i,%i)%f  \t", i, j, (i*smp.NColumnas) + j, smp.tabloide[(i*smp.NColumnas) + j]);
		}
		printf("\n");
	}
	
}

void printResult(TSimplexGPUs &smp) {
	printf("%s\n", "Resultado");
	double bi, val;
	double min = 0;
	int varType;
	
	for(int i = 0; i < smp.var_x; i++) {
		bi = findVarXbValue(smp, i);
		val = bi + smp.inf[i];
		if (val != 0) {
			printf("x%i = %.2f  (Xbi = %.2f)\n", findVarIndex(smp, i),  val, bi);
			min -= val * smp.z[i];
		}
	}
	
	for(int i = smp.var_x; i < smp.var_all; i++) {
		bi = findVarXbValue(smp, i);
		if (bi != 0) {
			varType = smp.var_type[i];
			if (varType == 1) {
				val = bi;
				printf("s%i = %.2f \n", findVarIndex(smp, i), val);
			} else {
				val = bi;
				printf("a - error%i = %.2f\n", findVarIndex(smp, i),  val);
			}
		}
	}
	
	printf("Z min = %.2f \n", min);
	
}

double findVarXbValue(TSimplexGPUs &smp, int indx) {
	int lefti;
	for(int i = 0; i < smp.rest_fin; i++) {
		lefti = ((int) smp.left[i*smp.mat_adv_row]);
		if (indx == (lefti - 1)) {
			return smp.Xb[i*smp.mat_adv_row];
		}
	}
	return 0;
}

int findVarIndex(TSimplexGPUs &smp, int indx) {
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





