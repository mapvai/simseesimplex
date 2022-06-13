#include <stdio.h>
#include <math.h>
#include <stdlib.h>		/* abs */
#include <assert.h>	/* assert */
#include <iostream>
#include <string>
using namespace std;

#include "toursimplexmgpus.h"

const double CasiCero_Simplex = 1.0E-7;
// const double AsumaCero =  1.0E-16; // EPSILON de la maquina en cuentas con Double, CONFIRMAR SI ESTO ES CORRECTO (double 64 bits, 11 exponente y 53 mantisa, 53 log10(2) ≈ 15.955 => 2E−53 ≈ 1.11 × 10E−16 => EPSILON  ≈ 1.0E-16)
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

const double M = 100; //100; //sqrt(MaxNReal);

void resolver_cpu(TSimplexGPUs &simplex) ;
void resolver_simplex_big_m(TSimplexGPUs &simplex) ;
bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol);
int locate_min_dj(TSimplexGPUs &smp);
int locate_min_ratio(TSimplexGPUs &smp, int zpos);
void resolver_ejemplo1();
void resolver_ejemplo2();
void resolver_ejemplo3();
void printStatus(TSimplexGPUs &smp);
void printResult(TSimplexGPUs &smp);
int findVarIndex(TSimplexGPUs &smp, int indx);

extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	/*
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		resolver_cpu(simplex_array[kTrayectoria]);
	}
	*/
	resolver_ejemplo3();
	
}

void resolver_ejemplo1() {
	
/*
	Problema Propuesto por Marco
		Min 4x1 +x2
		st:
		3x1 +1x2 = 3
		4x1 +3x2 ≥ 6
		1x1 +2x2 ≤ 4

		x1, x2 ≥ 0 .
=>		
		Min 4x1 +1x2 +MA1 +MA2
		Subject to:
		3x1 +x2 +A1 = 3
		4x1 +3x2 −s1 +A2 = 6
		x1 +2x2 +s2 = 4
		
		x1, x2, A1, s1, A2, s2 ≥ 0
Res:
	x1 = 0.400000, x2 = 1.800000, s1 = 1.000000 Verificado correcto
*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 6;
	simplex.NColumnas = simplex.NVariables + 3;
	simplex.NRestricciones = 3;
	simplex.Nfilas = simplex.NRestricciones + 3;
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
	// L, 		Cb, 	Xb
		0, 	0, 	0,		-4, 	-1, 	0, 	0, 	-M, 	-M, // z function
		0, 	0, 	0, 	0, 	0, 	1, 	1, 	2, 	2, // 0 var, 1 slack var, 2 artificial var
		0, 	0, 	0, 	3, 	4, 	5, 	6, 	7, 	8, // top vector
		-7,	-M,	3,	 	3, 	1, 	0, 	0, 	1,  	0,
		-8, 	-M, 	6, 	4, 	3,  	-1, 	0, 	0, 	1,
		-6, 	0, 	4, 	1, 	2, 	0, 	1, 	0, 	0
	};
	
	simplex.tabloide = (double*)&tabl;
	resolver_cpu(simplex);
}

void resolver_ejemplo2() {
	
/*
	Problema Propuesto por SimSEE
		Min x1 + 8x2 + 2x3
		st:
		x1 + x2 + x3 	≥ -10.5
		x1 + x2 			= 5.3
		x1  		- x3 		≤ 2.9

		0 ≤ x1 ≤ 12,  -6 ≤ x2 ≤ 6, -5 ≤ x3 ≤ 5
=>		
		Max -x1 - 8x2 - 2x3
		st:
		x1 + x2 + x3 	≥ -10.5
		x1 + x2 			= 5.3
		x1  		  - x3 	≤ 2.9
		x1					≤ 12
				x2			≤ 6
						x3	≤ 5
=> cambio variable para las cotas inferiores	
		Max -x1 - 8x2 - 2x3
		st:
		x1 + x2 + x3 	≥ -10.5 + 6 + 5 = 0.5
		x1 + x2 			= 5.3 + 6 		= 11.3
		x1  		  - x3 	≤ 2.9        - 5 	= - 2.1 => invert to get 2.1
		x1					≤ 12
				x2			≤ 12
						x3	≤ 10
						
		0 ≤ x1 ≤ 12,  0 ≤ x2 ≤ 12, 0 ≤ x3 ≤ 10				
=>
		Max -x1 - 8x2 - 2x3
		st:
		x1 + x2 + x3 	- s1 + a1 = 0.5
		x1 + x2 			+ a2 = 11.3
		-x1  		  + x3 	- s2 + a3 = 2.1
		x1					+ s3 = 12
				x2			+ s4 = 12
						x3	+ s5 = 10
=>
*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 11;
	simplex.NColumnas = simplex.NVariables + 3;
	simplex.NRestricciones = 6;
	simplex.Nfilas = simplex.NRestricciones + 3;
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
	// L, 		Cb, 	Xb
		0, 	0, 	0,			-1, 	-8, 	-2, 	0,	  -M, 	-M, 	0,		-M, 	0,		0,		0,	 // z function
		0, 	0, 	0, 		0, 	0, 	0, 	1, 	2, 	2, 	1, 	2, 	1, 	1, 	1, // 0 var, 1 slack var, 2 artificial var
		0, 	0, 	0, 		3, 	4, 	5, 	6, 	7, 	8,  	9, 	10, 	11, 	12, 	13,// top vector
		
		-7,	-M,	0.5,		1,		1,		1,		-1,	1,		0,		0,		0, 	0,		0,		0,
		-8, 	-M, 	11.3, 	1,		1,		0,		0,		0,		1,		0,		0, 	0,		0,		0,
		-10, 	-M, 	2.1, 		1,		0,		-1,	0,		0,		0,		-1,	1,		0,		0, 	0, 
		-11, 	0, 	12, 		1,		0,		0,		0,		0,		0,		0,		0, 	1,		0,		0,
		-12, 	0, 	12, 		0,		1,		0,		0,		0,		0,		0,		0, 	0,		1,		0,
		-13, 	0, 	10, 		0,		0,		1,		0,		0,		0,		0,		0, 	0,		0,		1
	};
	
	simplex.tabloide = (double*)&tabl;
	resolver_cpu(simplex);
}

void resolver_ejemplo3() {
	
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
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 11;
	simplex.NColumnas = simplex.NVariables + 3;
	simplex.NRestricciones = 6;
	simplex.Nfilas = simplex.NRestricciones + 3;
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
		0, 0, 0, -1, -3, -2, 0, -M, -M, 0, -M, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 
		0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
		7, -M, 0.5, 1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 
		8, -M, 0.7, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
		10, -M, 2.1, -1, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 
		11, 0, 12, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
		12, 0, 12, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
		13, 0, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1
	};
	
	simplex.tabloide = (double*)&tabl;
	resolver_cpu(simplex);
}


void resolver_cpu(TSimplexGPUs &simplex) {
	resolver_simplex_big_m(simplex);
}

void resolver_simplex_big_m(TSimplexGPUs &simplex) {
	int zpos, qpos, it;
	
	printf("resolver_simplex_big_m INT \n");
	
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
	for (z = 3; z < smp.NColumnas; z++) {
		top = smp.tabloide[2*smp.NColumnas + z] ;
		if (top > 0 && smp.tabloide[smp.NColumnas + top] != 2) { // it is not an artificial variable
			apz = -smp.tabloide[z];
			for (y = 3; y < smp.Nfilas; y++) {
				apz += smp.tabloide[y*smp.NColumnas + 1] * smp.tabloide[y*smp.NColumnas + z]; 
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
	for (y = 3; y < smp.Nfilas; y++) {
		denom = smp.tabloide[y*smp.NColumnas + zpos];
		printf("%.1f / %.1f ",  smp.tabloide[y*smp.NColumnas + 2], denom);
		// printf("Denominador: %f\n",  denom);
		if (denom > CasiCero_Simplex) {
			qy = smp.tabloide[y*smp.NColumnas + 2] / denom;
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

	double m, piv, invPiv;
	int i, j, ipos, k;

	piv = smp.tabloide[kfil * smp.NColumnas + jcol];
	invPiv = 1 / piv;
	
	ipos = kfil * smp.NColumnas;
	for (j = 2; j < smp.NColumnas; j++) {
		smp.tabloide[ipos + j] *= invPiv;
	}
	smp.tabloide[kfil * smp.NColumnas + jcol] = 1;
	
	for (i = 3; i < smp.Nfilas; i++) {
		if (i != kfil) {
			m = smp.tabloide[i *smp.NColumnas + jcol] ;
			for (j = 2; j < smp.NColumnas; j++) {
				if (j != jcol) {
					smp.tabloide[i *smp.NColumnas + j] -= m * smp.tabloide[kfil*smp.NColumnas + j]; 
				} else {
					smp.tabloide[i*smp.NColumnas + j] = 0;
				}
			}
		}
	}
	
	k = smp.tabloide[2*smp.NColumnas + jcol];
	smp.tabloide[2*smp.NColumnas + jcol] = smp.tabloide[smp.NColumnas*kfil];
	smp.tabloide[smp.NColumnas*kfil] = k;
	
	smp.tabloide[kfil*smp.NColumnas + 1] = smp.tabloide[jcol] ;
	
	return true;
 }


void printStatus(TSimplexGPUs &smp) {
	printf("%s, (%i, %i)\n", "Tabloide", smp.Nfilas, smp.NColumnas);
	for(int i = 0; i < smp.Nfilas; i++) {
		for(int j = 0; j < smp.NColumnas; j++) {
			printf("%.1f\t", smp.tabloide[i*smp.NColumnas + j] );
			//printf("%E \t", smp.tabloide[i*smp.NColumnas + j] );
			//printf("(%i,%i,%i)%f  \t", i, j, (i*smp.NColumnas) + j, smp.tabloide[(i*smp.NColumnas) + j]);
		}
		printf("\n");
	}
	
}

void printResult(TSimplexGPUs &smp) {
	printf("%s\n", "Resultado");
	int indx;
	string nvar = "xos";
	
	/* // Print left vector
	printf("left = [%f", smp.tabloide[3*smp.NColumnas]);
	for(int i = 4; i < smp.Nfilas; i++) printf(", %f", smp.tabloide[i*smp.NColumnas]);
	printf("]\n");
	*/
	
	//printf("z = %f\n", smp.tabloide[smp.NVariables]);
	int varType;
	for(int i = 3; i < smp.Nfilas; i++) {
		indx = smp.tabloide[i*smp.NColumnas];
		indx = (indx < 0)? -indx: indx;
		varType = smp.tabloide[smp.NColumnas + indx];
		if (varType == 0) {
				nvar = "x";
		} else if (varType == 1) {
			nvar = "s";
		} else {
			nvar = "a - error ";
		}
		
		printf("%s%i = %.2f \n", nvar.c_str(), findVarIndex(smp, indx),  smp.tabloide[i*smp.NColumnas + 2]);
	}
	
}

int findVarIndex(TSimplexGPUs &smp, int indx) {
	int vind = 1;
	int varType = smp.tabloide[smp.NColumnas + indx];
	if ( varType == 0) {
		return indx - 2;
	} else {
		for (int i = 3; i < indx; i++) {
			if (smp.tabloide[smp.NColumnas + i] == varType) vind++;
		}
	}
	return vind;
}





