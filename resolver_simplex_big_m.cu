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
int locate_zpos(TSimplexGPUs &smp);
int locate_qpos(TSimplexGPUs &smp, int zpos);
void resolver_ejemplo1();
void resolver_ejemplo2();
void printStatus(TSimplexGPUs &smp);
void printResult(TSimplexGPUs &smp);

extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	/*
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		resolver_cpu(simplex_array[kTrayectoria]);
	}
	*/
	resolver_ejemplo2();
	
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
=>
*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 6;
	simplex.NColumnas = simplex.NVariables + 3;
	simplex.NRestricciones = 3;
	simplex.Nfilas = simplex.NRestricciones + 3;
	simplex.cantArtVars = 2;
	
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
=>
*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 6;
	simplex.NColumnas = simplex.NVariables + 3;
	simplex.NRestricciones = 3;
	simplex.Nfilas = simplex.NRestricciones + 3;
	simplex.cantArtVars = 2;
	
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


void resolver_cpu(TSimplexGPUs &simplex) {
	resolver_simplex_big_m(simplex);
}

void resolver_simplex_big_m(TSimplexGPUs &simplex) {
	int zpos, qpos, it;
	
	printf("resolver_simplex_big_m INT \n");
	
	printStatus(simplex);
	it = 0;
	
	do {
		zpos = locate_zpos(simplex);
		printf("%s %d \n", "zpos", zpos);
		
		if (zpos < 0) {
			printf("%s\n", "Condicion de parada maximo encontrado");
			printResult(simplex);
			return;
		}
		
		qpos = locate_qpos(simplex, zpos);
		printf("%s %d \n", "qpos", qpos);
		if (qpos < 0) {
			printf("%s\n", "Posicion de cociente no encontrada");
			return;
		}
		
		intercambiarvars(simplex, qpos, zpos);
		
		printStatus(simplex);
		
		it++;
		
		if (it == 4) {
			printf("Max %i iterations achieved\n", it);
			return;
		}
	} while (true);
	
}

int locate_zpos(TSimplexGPUs &smp) {
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


int locate_qpos(TSimplexGPUs &smp, int zpos) {
	int mejory, y;
	double min_apy, qy;

	mejory = -1;
	min_apy = MaxNReal;
	for (y = 3; y < smp.Nfilas; y++) {
		qy = smp.tabloide[y*smp.NColumnas + 2] / smp.tabloide[y*smp.NColumnas + zpos];
		if (qy > 0 && qy < min_apy) {
			mejory = y;
			min_apy = qy;
			printf("MIn Q: %f\n",  min_apy);
		}
	}
	
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
			printf("%f \t", smp.tabloide[i*smp.NColumnas + j] );
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
	
	//printf("left = [%i", smp.left[0]);
	for(int i = 3; i < smp.Nfilas; i++) printf(", %f", smp.tabloide[i*smp.Nfilas]);
	printf("]\n");
	
	//printf("z = %f\n", smp.tabloide[smp.NVariables]);
	
	for(int i = 3; i < smp.Nfilas; i++) {
		indx = smp.tabloide[i*smp.NColumnas];
		indx = (indx < 0)? -indx: indx;
		if (smp.tabloide[smp.NColumnas + indx] == 0) {
				nvar = "x";
		} else {
			nvar = "s";
		}
		
		printf("%s%i = %f \n", nvar.c_str(), indx - 2,  smp.tabloide[i*smp.NColumnas + 2]);
	}
	
}





