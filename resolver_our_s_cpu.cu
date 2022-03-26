#include <stdio.h>
#include <math.h>
#include <stdlib.h>		/* abs */
#include <assert.h>	/* assert */
#include <iostream>
#include <string>
using namespace std;

#include "toursimplexgpus.h"

const double CasiCero_Simplex = 1.0E-7;
// const double AsumaCero =  1.0E-16; // EPSILON de la maquina en cuentas con Double, CONFIRMAR SI ESTO ES CORRECTO (double 64 bits, 11 exponente y 53 mantisa, 53 log10(2) ≈ 15.955 => 2E−53 ≈ 1.11 × 10E−16 => EPSILON  ≈ 1.0E-16)
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO

void resolver_cpu(TSimplexGPUs &simplex) ;
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
	resolver_ejemplo1();
	resolver_ejemplo2();
	
}

void resolver_ejemplo1() {
	
	/*
	Problema Propuesto por Cami
	  max z = 7x1 + 4x2
	  s.a.
		2x1 + x2   <= 20
		x1  + x2   <= 18
		x1         <= 8  
	*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 5;
	simplex.NRestricciones = 3;
	
	simplex.top = (int*)malloc((simplex.NVariables)*sizeof(int));
	for (int i = 0; i < simplex.NVariables; i++) simplex.top[i] = i +1;
	
    simplex.left = (int*)malloc((simplex.NRestricciones)*sizeof(int));
	for (int i = 0; i < simplex.NRestricciones; i++) simplex.left[i] = i +1;
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
		-7, -4, 0, 0, 0, 0, 
		2, 1, 1, 0, 0, 20,
		1, 1, 0, 1, 0, 18,
		1, 0, 0, 0, 1, 8,
	};
	
	simplex.tabloide = (double*)&tabl;
	resolver_cpu(simplex);
}

void resolver_ejemplo2() {
	
	/*
	Problema Propuesto por Cami, mas que agrego una restriccion de igualdad, con 0 en las slack variables
	  max z = 7x1 + 4x2
	  s.a.
		2x1 + x2   <= 20
		x1  + x2   <= 18
		x1         <= 8  
	*/
	
	TSimplexGPUs simplex; // (TSimplexGPUs*)malloc(sizeof(TSimplexGPUs));
	simplex.NVariables = 5;
	simplex.NRestricciones = 4;
	
	simplex.top = (int*)malloc((simplex.NVariables)*sizeof(int));
	for (int i = 0; i < simplex.NVariables; i++) simplex.top[i] = i + 1;
	
    simplex.left = (int*)malloc((simplex.NRestricciones)*sizeof(int));
	for (int i = 0; i < simplex.NRestricciones; i++) simplex.left[i] = i - 1;
	
    //simplex->tabloide = (double*)malloc((simplex->NVariables + 1)*(simplex->NRestricciones + 1)*sizeof(double));
	double tabl[] = {
		-7, -4, 0, 0, 0, 0, 
		2, 1, 1, 0, 0, 20,
		1, 1, 0, 1, 0, 18,
		1, 0, 0, 1, 0, 8,
		0, 1, 0, 0, 0, 16
	};
	
	simplex.tabloide = (double*)&tabl;
	resolver_cpu(simplex);
}


void resolver_cpu(TSimplexGPUs &simplex) {
	int zpos, qpos, it;
	
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
		if (it == 4) return;
		
	} while (true);
	
}


int locate_zpos(TSimplexGPUs &smp) {
	int mejorz, z;
	double min_apz, apz;

	mejorz = -1;
	min_apz = 0;
	for (z = 0; z < smp.NVariables; z++) {
		apz = smp.tabloide[z];
		if (apz < 0 && apz < min_apz) {
			mejorz = z;
			min_apz = apz;
		}
	}
	
	return mejorz;
}


int locate_qpos(TSimplexGPUs &smp, int zpos) {
	int mejorq, q;
	double min_apq, apq, qz;

	mejorq = -1;
	min_apq = MaxNReal;
	for (q = 1; q <= smp.NRestricciones; q++) {
		qz = smp.tabloide[q * (smp.NVariables + 1) + zpos]; // [q][zpos]
		if (qz > CasiCero_Simplex) { // > 0
			apq = smp.tabloide[q * (smp.NVariables + 1) + smp.NVariables]/qz;
			if (apq < min_apq) {
				mejorq = q;
				min_apq = apq;
			}
		}
	}
	
	return mejorq;
	
}

bool intercambiarvars(TSimplexGPUs &smp, int kfil, int jcol) {

	double m, piv, invPiv;
	int i, j, ipos, k;

	piv = smp.tabloide[kfil * (smp.NVariables + 1) + jcol];
	invPiv = 1 / piv;
	
	ipos = kfil * (smp.NVariables + 1) ;
	for (j = 0; j <= smp.NVariables; j++) {
		smp.tabloide[ipos + j] *= invPiv;
	}
	
	for (i = 0; i <= smp.NRestricciones; i++) {
		if (i != kfil) {
			m = smp.tabloide[i * (smp.NVariables + 1) + jcol] ;
			for (j = 0; j <= smp.NVariables; j++) {
				if (j != jcol) {
					smp.tabloide[i * (smp.NVariables + 1) + j] -= m * smp.tabloide[kfil * (smp.NVariables + 1) + j]; 
				} else {
					smp.tabloide[i * (smp.NVariables + 1) + j] = 0;
				}
			}
		}
	}
	
	for (i = 0; i <= smp.NRestricciones; i++) {
		if (i != kfil) {
			smp.tabloide[i * (smp.NVariables + 1) + jcol] /= -invPiv; 
		} else {
			smp.tabloide[i * (smp.NVariables + 1) + jcol]  = invPiv;
		}
	}
	
	k = smp.top[jcol];
	smp.top[jcol] = smp.left[kfil];
	smp.left[kfil] = k;

	return true;
 }


void printStatus(TSimplexGPUs &smp) {
	printf("%s\n", "Tabloide");
	for(int i = 0; i <= smp.NRestricciones; i++) {
		for(int j = 0; j <= smp.NVariables; j++) {
			printf("%f ", smp.tabloide[i* (smp.NVariables + 1) + j] );
		}
		printf("\n");
	}
	
}

void printResult(TSimplexGPUs &smp) {
	printf("%s\n", "Resultado");
	int indx;
	string nvar = "xos";
	int nobvari = smp.NVariables - smp.NRestricciones;
	
	printf("left = [%i", smp.left[0]);
	for(int i = 1; i <= smp.NRestricciones; i++) printf(", %i", smp.left[i]);
	printf("]\n");
	
	printf("z = %f\n", smp.tabloide[smp.NVariables]);
	
	for(int i = 1; i <= smp.NRestricciones; i++) {
		if (smp.left[i] > 0) {
			if (smp.left[i] <= nobvari) {
				nvar = "x";
				indx = smp.left[i];
			} else {
				nvar = "s";
				indx = smp.left[i] - nobvari;
			}
		} else {
			nvar = "s";
			indx = -smp.left[i];
		}
		
		printf("%s%i = %f \n", nvar.c_str(), indx,  smp.tabloide[i* (smp.NVariables + 1) + smp.NVariables]);
	}
	
}





