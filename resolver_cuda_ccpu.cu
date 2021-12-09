#include <stdio.h>
#include <math.h>
#include <stdlib.h>     /* abs */


#include "tsimplexgpus.h"


const double CasiCero_Simplex_Escalado = 1.0E-30;
const double CasiCero_Simplex = 1.0E-7;
const double CasiCero_Simplex_CotaSup = CasiCero_Simplex * 1.0E+3;
const double CasiCero_VarEntera = 1.0E-5;
const double CasiCero_CajaLaminar = 1.0E-30;
const double AsumaCero =  1.0E-16; // EPSILON de la maquina en cuentas con Double, CONFIRMAR SI ESTO ES CORRECTO (double 64 bits, 11 exponente y 53 mantisa, 53 log10(2) ≈ 15.955 => 2E−53 ≈ 1.11 × 10E−16 => EPSILON  ≈ 1.0E-16)
const double MaxNReal = 1.7E+308; // Aprox, CONFIRMAR SI ESTO ES CORRECTO


extern "C" void resolver_cuda(TDAOfSimplexGPUs &simplex_array, TDAOfSimplexGPUs &d_simplex_array, TDAOfSimplexGPUs &h_simplex_array, int NTrayectorias) {
	int NEnteras;
	int NVariables;
	int NRestricciones;
	int cnt_varfijas;
	int cnt_RestriccionesRedundantes;
	
	//TSimplexVars smp_var_arr = (SimplexVars*)malloc(NTrayectorias*sizeof(SimplexVars));
	
	for (int kTrayectoria = 0; kTrayectoria < NTrayectorias; kTrayectoria++) {
		resolver_cpu(&simplex_array[kTrayectoria]); //,  &smp_var_arr[kTrayectoria]); 
	}
	
}


void resolver_cpu(TDAOfSimplexGPUs &simplex) { //,  TSimplexVars &vars) {
	int res;
	int cnt_columnasFijadas = 0; // Cantidad de columnas FIJADAS x o y fija encolumnada
	int result = 0;
	int cnt_varfijas = 0; // Cantidad de variables fijadas
	string mensajeDeError;

	fijarCajasLaminares(simplex, cnt_varfijas);

	// Fijamos las variables que se hayan declarado como constantes.
	if (!fijarVariables(simplex, cnt_varfijas, cnt_columnasFijadas)) {
		mensajeDeError = 'PROBLEMA INFACTIBLE - No fijar las variable Fijas.';
		result = -32;
		return;
	}

	if (!enfilarVariablesLibres(simplex, cnt_columnasFijadas, cnt_RestriccionesRedundantes, cnt_VariablesLiberadas)) {
		mensajeDeError = 'No fue posible conmutar a filas todas las variables libres';
		result = -33;
		return;
	}

	if (resolverIgualdades(simplex, cnt_columnasFijadas, cnt_varfijas, cnt_RestriccionesRedundantes) != 1) {
		mensajeDeError = 'PROBLEMA INFACTIBLE - No logré resolver las restricciones de igualdad.';
		result = -31;
		return;
	}

	lbl_inicio:

	reordenarPorFactibilidad(simplex, cnt_RestriccionesRedundantes, cnt_RestrInfactibles); // MAP: cnt_RestrInfactibles es modificada dentro del proc

	res = 1;
	while (cnt_RestrInfactibles > 0) {
    res = pasoBuscarFactible(simplex, cnt_RestrInfactibles);

    switch (res) {
		case 0: 
			if (cnt_RestrInfactibles > 0) {
				mensajeDeError := 'PROBLEMA INFACTIBLE - Buscando factibilidad';
				result = -10;
				return;
			}
			break;
		case  -1:
			mensajeDeError = 'NO encontramos pivote bueno - Buscando Factibilidad';
			result = -11;
			return;
		case -2:
			mensajeDeError = '???cnt_infactibles= 0 - Buscando Factibilidad';
			result = -12;
			return;
	}

	while (res == 1) {
		res = darpaso(simplex);
		if (res == -1) {
			mensajeDeError = 'Error -- NO encontramos pivote bueno dando paso';
			result = -21;
			return;
		}
	}
  
	if (res == 2) {
		goto lbl_inicio;
	}
	
	result = res;
}


// Si abs( cotasup - cotainf ) < AsumaCeroCaja then flg_x := 2
// retorna la cantidad de cajas fijadas.
int fijarCajasLaminares(TDAOfSimplexGPUs &smp, int &cnt_varfijas) {
	int res = 0;
	
	for (int i =  0; i < smp.NVariables; i++) { // MAP: = for 1 to nc-1 
		if (abs(smp.flg_x[i] ) < 2) { // No se aplica ni para 2 ni para 3
			if (abs(smp.x_sup[i] - smp.x_inf[i] ) < CasiCero_CajaLaminar) {
				if (smp.flg_x[i] < 0) {
					smp.flg_x[i] = -2;
				} else {
					smp.flg_x[i] = 2;
				}
				cnt_varfijas++;
				res++;
			}
		}
	}
    return res;
}


void posicionarPrimeraLibre(TDAOfSimplexGPUs &smp, int cnt_varfijas, int &cnt_fijadas, int &cnt_columnasFijadas, int &kPrimeraLibre) {
    while (cnt_fijadas < cnt_varfijas) &&
      (((smp.top[kPrimeraLibre] < 0) && (abs(smp.flg_x[-smp.top[kPrimeraLibre]]) == 2)) or
        ((smp.top[kPrimeraLibre] > 0) && (abs(smp.flg_y[smp.top[kPrimeraLibre]]) == 2))) {
		if (smp.top[kPrimeraLibre] < 0) {
			cnt_fijadas++;
		}	
		cnt_columnasFijadas++;
		kPrimeraLibre--;
	}
}


bool fijarVariables(TDAOfSimplexGPUs &smp, int cnt_varfijas, int &cnt_columnasFijadas) {

	int kColumnas, mejorColumnaParaCambiarFila, kFor, kFilaAFijar, cnt_fijadas, kPrimeraLibre;
	double mejorAkFilai;
	bool buscando, pivoteoConUnaFijada;

	if (cnt_varfijas > 0) {
		cnt_fijadas = 0;
		kPrimeraLibre = smp.NVariables - 1; // MAP: nc - 1, - 1 debido al cambio de indices
		kColumnas = 0; // MAP: Antes 1, pero cambimos que los vectores esten indexados desde 1 a 0
		
		while ((cnt_fijadas < cnt_varfijas) && (kColumnas <= kPrimeraLibre)) {
			posicionarPrimeraLibre(smp, cnt_varfijas, cnt_columnasFijadas);
			//Busco en columnas
			if ((cnt_fijadas < cnt_varfijas) && (kColumnas <= kPrimeraLibre)) {
				buscando = true;
				while (buscando && (kColumnas <= kPrimeraLibre)) {
					
					// MAP: Aca grego -1 a flg_x[-top[kColumnas] -1] para obtener el indice que empieza en 0 y top left son 2 vectores indexados en 0 pero que contienen numeros comenzados en 1, por lo tanto el indice real es top[indice] - 1,
					// esto lo voy a tener que hacer a lo largo y ancho del algoritmo
					if ((smp.top[kColumnas]) < 0) && (abs(smp.flg_x[-smp.top[kColumnas] - 1]) == 2) ) { 
						//es una x fija
						buscando = false
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
		
			posicionarPrimeraLibre(smp, cnt_varfijas, cnt_columnasFijadas);
			if (cnt_fijadas < cnt_varfijas) {
		 
				//Busco en filas
				for (kFor = kFilaAFijar + 1; kFor < smp.NVariables - 1; kFor++) { // MAP: change index in loop
					if ((smp.left[kFor] < 0) && (abs(smp.flg_x[-smp.left[kFor] - 1]) == 2)) {
						kFilaAFijar = kFor;
						break;
					}
				}

				mejorColumnaParaCambiarFila = 0; // MAP: antes 1
				mejorAkFilai = abs(smp.mat[kFilaAFijar][0]); // MAP: before [1], primera columna ahora es 0
				for (kColumnas = 1; kColumnas < kPrimeraLibre; kColumnas++) { // MAP: Antes 2 to kPrimeraLibre
					if (abs(smp.mat[kFilaAFijar][kColumnas]) > mejorAkFilai) {
						mejorColumnaParaCambiarFila = kColumnas;
						mejorAkFilai = abs(smp.mat[kFilaAFijar][kColumnas]);
					}
				}

				pivoteoConUnaFijada = false;
				// dv@20191226 Si el término independiente es nulo, la "restricción" es reduntante
				// entonces la variable ya había quedado fijada
				if (mejorAkFilai < AsumaCero) {
					return false;
				}

				intercambiar(smp, kFilaAFijar, mejorColumnaParaCambiarFila);
				cnt_fijadas++;
				
				if (!pivoteoConUnaFijada) {
					// dv@20200115 agrego esto porque no debería cambiar si pivoteó con una fijada
					if (mejorColumnaParaCambiarFila != kPrimeraLibre) {
						intercambioColumnas(smp, mejorColumnaParaCambiarFila, kPrimeraLibre);
					}
					cnt_columnasFijadas++;
					kPrimeraLibre--;
				} else {
					if (mejorColumnaParaCambiarFila != kPrimeraLibre + 1) { // En el caso de que se pivotee con una columna fija no cambia kPrimeraLibre
						intercambioColumnas(smp, mejorColumnaParaCambiarFila, kPrimeraLibre + 1);
					}
				}
			}
		}
	}
	
	return true;

}


bool intercambiar(TDAOfSimplexGPUs &smp, int kfil, int jcol) {

	double m, piv, invPiv;
	int k, j;

	piv := smp.mat[kfil][jcol];
	invPiv := 1 / piv;

	// MAP: POSIBLE MEJORA, en GPU se encargaran hilos diferentes, recorrida por filas
	// MAP: En el codigo original se separa en 4 casos el mismo codigo, se recorre desde 1 to cnt_RestriccionesRedundantes to kfil - 1  <skipping kfil (fila pivot)> to nf - 1 to nf
	// MAP: Se concatenan estos casos en un mismo for
	for (k = 0; k < kfil; k++) {
		m := -smp.mat[k][jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k][j] = smp.mat[k][j] + m * smp.mat[kfil][j]; 
			}
			
			smp.mat[k][jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k][j] = smp.mat[k][j] + m * smp.mat[kfil][j];
			}
		} else {
			smp.mat[k][jcol] = 0; // IMPONGO_CEROS
		}
	}
	
	// Salteo la fila kfil
	
	for (k = kfil + 1; k <= smp.NRestricciones; k++) { // MAP: <= pq considera la fila de la funcion z
		m := -smp.mat[k][jcol] * invPiv;
		if (abs(m) > 0) {
			for (j = 0; j < jcol; j++) {
				smp.mat[k][j] = smp.mat[k][j] + m * smp.mat[kfil][j]; 
			}
			
			smp.mat[k][jcol] = -m;
			
			for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes "to nc"
				smp.mat[k][j] = smp.mat[k][j] + m * smp.mat[kfil][j];
			}
		} else {
			smp.mat[k][jcol] = 0; // IMPONGO_CEROS
		}
	}

	// Completo la fila kfil
	m := -invPiv;
	for (j = 0; j < jcol; j++) {
		smp.mat[kfil][j] = smp.mat[kfil][j] * m;
	}

	smp.mat[kfil][jcol] = -m;
	
	for (j = jcol + 1; j <= smp.NVariables; j++) { // MAP: Antes jcol + 1 to nc
		smp.mat[kfil][j] = smp.mat[kfil][j] * m;
	}

	k = smp.top[jcol];
	smp.top[jcol] = smp.left[kfil];
	smp.left[kfil] = k;

	actualizo_iitop(jcol);
	actualizo_iileft(kfil);

  return true;
 }

// MAP: Agrego los -1 +1 por el cambio de indices CHEQUEAR QUE ES CORRECTO
void actualizo_iitop(TDAOfSimplexGPUs &smp, int k) {
	// actualizo los indices iix e iiy
	if (smp.top[k] < 0) {
		smp.iix[-smp.top[k] - 1] = k + 1;
	} else {
		smp.iiy[smp.top[k] - 1] = -k - 1;
	}
}

// MAP: Agrego los -1 +1 por el cambio de indices CHEQUEAR QUE ES CORRECTO
void actualizo_iileft(TDAOfSimplexGPUs &smp, int k) {
	// actualizo los indices iix e iiy
	if (smp.left[k] > 0) {
		smp.iiy[smp.left[k] - 1]  = k + 1; 
	} else {
		smp.iix[-smp.left[k] - 1]  = -k - 1;
	}
}

void intercambioColumnas(TDAOfSimplexGPUs &smp, int j1, int j2) {

	int k;
	double m;

	for (k = 0; k <= smp.NRestricciones; k++) { // MAP: Intercambia toda la columna por lo tanto va hasta smp.NRestricciones inclusive
		m = smp.mat[k][j1];
		smp.mat[k].pv[j1]  = smp.mat[k][j2];
		smp.mat[k].pv[j2]  = m;
	}

	k = smp.top[j1];
	smp.top[j1] = smp.top[j2];
	smp.top[j2] = k;

	actualizo_iitop(smp, j1);
	actualizo_iitop(smp, j2);
}

void intercambioFilas(TDAOfSimplexGPUs &smp, int k1, int k2) {
  
	int j;
	double m;

	for (j = 0; j <= smp.NVariables; j++) { // MAP: Intercambia toda la fila por lo tanto va hasta smp.NVariables inclusive
		m = smp.mat[k1][j];
		smp.mat[k1].pv[j] = smp.mat[k2][j];
		smp.mat[k2].pv[j] = m;
	}

	j = smp.left[k1];
	smp.left[k1] = smp.left[k2];
	smp.left[k2] = j;

	actualizo_iileft(smp, k1);
	actualizo_iileft(smp, k2);

}

 // Atención esto funciona porque suponemos que el Simplex está en su estado Natural.
 // a lo sumo se conmutaron algunas columnas para fijar variables.
 // MAP: se usa solo dentro de enfilarVariablesLibres
 int buscarMejorPivoteEnCol(TDAOfSimplexGPUs &smp, int jCol, int iFilaFrom, int iFilaHasta) {
	
	double a;
	int iFil, res;
	
    res = -1;
    a = 0.0;
	for (iFil = iFilaFrom; iFil <= iFilaHasta; iFil++) { // MAP: Originalmente iFilaFrom to iFilaHasta, eso no cambia ya que se le pasan los indices ya adaptados a la indexacion desde 0
		if (abs(smp.mat[iFil][jCol]) > a) {
			a = abs(smp.mat[iFil][jCol]);
			res = iFil;
		}
	}
	return res;
 }


bool enfilarVariablesLibres(TDAOfSimplexGPUs &smp, int cnt_columnasFijadas, int &cnt_RestriccionesRedundantes, int &cnt_VariablesLiberadas) {

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
			if (iFil > cnt_restriccionesRedundantes) {
				IntercambioFilas(smp, cnt_RestriccionesRedundantes, iFil);
			}
			cnt_VariablesLiberadas++;
		}
	}
  
	return true;
}


int resolverIgualdades(TDAOfSimplexGPUs &smp, int &cnt_columnasFijadas, int cnt_varfijas, int &cnt_RestriccionesRedundantes) {
	int res, cnt_acomodadas, ifila, icolumna, 
		nIgualdadesResueltas, nIgualdadesAResolver, 
		iFilaLibre, iFilaAcomodando;
	int * nCerosFilas;
	int * nCerosCols;
	bool fantasma;

	cnt_acomodadas = cnt_columnasFijadas - cnt_varfijas;

	// Muevo las igualdades que esten en columnas al lado derecho junto con las FIJADAS
	icolumna = smp.NVariables - cnt_columnasFijadas - 1; // MAP: Indice modificado, cambio nc por smp.NVariables 
	while ((icolumna >= 0) && (cnt_acomodadas < cnt_Igualdades)) {
		if ((smp.top[icolumna] > 0) && (abs(smp.flg_y[top[icolumna] - 1]) == 2)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0
			if (icolumna <> (nc - cnt_columnasFijadas - 1)) {
				IntercambioColumnas(icolumna, nc - cnt_columnasFijadas - 1);
			}
			cnt_acomodadas++;
			cnt_columnasFijadas++;
		}
		icolumna--;
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
	// redundantes. Esto hacía que el  "while  cnt_acomodadas < cnt_Igualdades"
	// que está unas lineas abajo NO saliera por no alcanzar la condición.
	// rch@20130307.bugfix - end ----------------------------


	// ahora reordeno las igualdades y las que queden en filas las pongo al inicio
	iFilaLibre = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0
	iFilaAcomodando = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0

	while (cnt_acomodadas < cnt_Igualdades) {
		if ((smp.left[iFilaAcomodando] > 0) and (abs(smp.flg_y[smp.left[iFilaAcomodando] - 1]) == 2)) { // MAP: Agrego - 1 para mover el indice a la indexacion desde 0
			//vc, dv @20200116 decia flg_y[iFilaAcomodando], se fijaba en cualquier lado
			//  Es una restricción  y es de igualdad
			if (iFilaLibre <> iFilaAcomodando) {
				IntercambioFilas(iFilaAcomodando, iFilaLibre);
			}
			cnt_acomodadas++;
			iFilaLibre++;
		}
		iFilaAcomodando++;
	} //Al salir de aca iFilaLibre queda en la primer fila que no es de igualdad

	res = 1;

	nIgualdadesResueltas = 0;
	nIgualdadesAResolver = iFilaLibre - (cnt_RestriccionesRedundantes + 1);
	nCerosFilas = (int*)malloc((smp.NRestricciones + 1)*sizeof(int)); // MAP: antes setLength(nCerosFilas, nf);
	nCerosCols = (int*)malloc((smp.NVariables + 1)*sizeof(int));// MAP: antes setLength(nCerosCols, nc);
	while (nIgualdadesResueltas < nIgualdadesAResolver) {
		//    res:= pasoBuscarFactibleIgualdad( cnt_RestriccionesRedundantes + 1 + nIgualdadesResueltas );
		//    res:= pasoBuscarFactibleIgualdad2( cnt_RestriccionesRedundantes + 1 + nIgualdadesResueltas );
		//    res:= pasoBuscarFactibleIgualdad3( nIgualdadesAResolver - nIgualdadesResueltas);
		res = pasoBuscarFactibleIgualdad4(nIgualdadesAResolver - nIgualdadesResueltas, nCerosFilas, nCerosCols, cnt_columnasFijadas, cnt_RestriccionesRedundantes);

		if (res = 1) {
			nIgualdadesResueltas = nIgualdadesResueltas + 1;
			cnt_columnasFijadas++;
		} else {
			ifila = cnt_RestriccionesRedundantes; // MAP: remuevo el +1 debido a la nueva indexacion desde 0
			while ((nIgualdadesResueltas < nIgualdadesAResolver) && filaEsFactible(smp, ifila, fantasma)) {
				if (iFila <> cnt_RestriccionesRedundantes + 1) {
					IntercambioFilas(ifila, cnt_RestriccionesRedundantes + 1);
				}
				cnt_RestriccionesRedundantes++;
				nIgualdadesResueltas++;
				ifila++;
			}
			if (nIgualdadesResueltas < nIgualdadesAResolver) {
				mensajeDeError = 'PROBLEMA INFACTIBLE - Resolviendo igualdades.';
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
bool filaEsFactible(TDAOfSimplexGPUs &smp, int kfila, bool &fantasma) {
	
	int ix;
	// if e(kfila, nc) < -CasiCero_Simplex then
	if (smp.mat[kfila][smp.NVariables] < -CasiCero_Simplex) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		// Si la fila es < 0
		fantasma = false;
		return false;
	else if (smp.left[kfila] < 0) then
		// Si rval es >= 0 reviso si es una variable con cota superior
		ix = -smp.left[kfila];
		// if (flg_x[ix] <> 0) and (e(kfila, nc) > (x_sup.pv[ix] + CasiCero_Simplex_CotaSup)) then
		if ((smp.flg_x[ix - 1] <> 0) && ((smp.mat[kfila][smp.NVariables]) > (smp.x_sup.pv[ix - 1] + CasiCero_Simplex_CotaSup))) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice y agrego -1 para correr el indice
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


int pasoBuscarFactibleIgualdad4(TDAOfSimplexGPUs &smp, int IgualdadesNoResueltas, int * nCerosFilas, int * nCerosCols, int cnt_columnasFijadas, int cnt_RestriccionesRedundantes) {

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
			m = abs(smp.mat[iFila][iColumna]);
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
	for (iFila := cnt_RestriccionesRedundantes + nIgualdadesNoResueltas; iFila < smp.NRestricciones; iFila++) { // MAP: cnt_RestriccionesRedundantes + nIgualdadesNoResueltas + 1 to nf - 1
		for (iColumna = 0; iColumna < columnasLibres - 1; iColumna++) { // MAP: Originalmente 1 to columnasLibres - 1
			if (abs(smp.mat[iFila][iColumna]) < AsumaCero) {
				nCerosCols[iColumna]++;
			}
		}
	}

	if (maxVal > CasiCero_Simplex) {
		for iFila := cnt_RestriccionesRedundantes + 1 to cnt_RestriccionesRedundantes +	nIgualdadesNoResueltas) { // MAP: Originalmente cnt_RestriccionesRedundantes + 1 to cnt_RestriccionesRedundantes +	nIgualdadesNoResueltas
			for (iColumna = 0; iColumna < columnasLibres; iColumna++) { // MAP: Originalmente 1 to columnasLibres
				if (abs(smp.mat[iFila][iColumna]) * 10 >= maxVal) {
					// Lo considero como posible pivote
					if (nCerosFilas[filaPiv] + nCerosCols[colPiv]) < (nCerosFilas[iFila] + nCerosCols[iColumna]) {
						filaPiv = iFila;
						colPiv = iColumna;
					}
				}
			}
		}

		// Muevo la fila a intercambiar al final asi me siguen quedando las que voy a acomodar en bloque desde cnt_RestriccionesRedundantes
		if (filaPiv <> cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1) { // MAP: Agrego -1 para correr el indice, filaPiv ya esta en el indice correcto
			IntercambioFilas(filaPiv, cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1); // MAP: Agrego -1 para correr el indice, filaPiv ya esta en el indice correcto
		}
		intercambiar(cnt_RestriccionesRedundantes + nIgualdadesNoResueltas - 1, colPiv); // MAP: Agrego -1 para correr el indice, colPiv ya esta en el indice correcto
		if (colPiv <> columnasLibres) {
			IntercambioColumnas(colPiv, columnasLibres - 1); // MAP: Agrego -1 para correr el indice, colPiv ya esta en el indice correcto
		}
		return 1;
	} else {
		return -1;
	}	
}

int reordenarPorFactibilidad(TDAOfSimplexGPUs &smp, int cnt_RestriccionesRedundantes, int &cnt_RestrInfactibles) {

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
    rval = smp.mat[kfil][nc];
    if (rval > 0) {
      // Si es = 0 no chequeo pues la fantasma no puede estar violada
		if (smp.left[kfil] < 0) {
			ix = -smp.left[kfil] - 1; // MAP: Agrego -1 para ajustar el indice a la indexacion desde 0
			if (smp.flg_x[ix] <> 0) and (x_sup[ix] < rval) then
				// Parece que violo la cota superior
				if ((smp.x_sup[ix] + CasiCero_Simplex_CotaSup) < rval) {
					// La viola realmente
					cambiar_borde_de_caja(smp, kfil);
				} else {
					// La viola por errores númericos
					// pon_e(kfil, nc, x_sup.pv[ix])
					smp.mat[kfil][smp.NVariables] = smp.x_sup[ix]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				}
			}
		}
    } else {
		if (rval > -CasiCero_Simplex_CotaSup) {
		  // pon_e(kfil, nc, 0);
		  smp.mat[kfil][smp.NVariables] = 0; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		}
	}

	// Ahora sabemos que las violadas están explícitas, movemos todas las
	//restricciones violadas al final
	kfil = cnt_RestriccionesRedundantes; // MAP: Remuevo + 1 para ajustar el indice a la indexacion desde 0
	cnt_RestrInfactibles = 0;
	while (kfil < (smp.NRestricciones - cnt_RestrInfactibles)) { // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
		// rval:= e(kfil, nc);
		rval = smp.mat[kfil][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		if (rval < 0) {
			Icnt_RestrInfactibles++;
			// while (e(nf-cnt_RestrInfactibles, nc ) < 0)
			// MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice, idem para nc	
			while ((smp.mat[smp.NRestricciones - cnt_RestrInfactibles][smp.NVariables] < 0) && (kfil < (smp.NRestricciones - cnt_RestrInfactibles))) {
				cnt_RestrInfactibles++;
			}
			if (kfil < (smp.NRestricciones - cnt_RestrInfactibles)) { // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
				IntercambioFilas(kfil, smp.NRestricciones - cnt_RestrInfactibles); // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
			}
		}
		kfil++;
	}
	return cnt_RestrInfactibles;
}

void cambiar_borde_de_caja(TDAOfSimplexGPUs &smp, int k_fila) {
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
		smp.mat[k_fila][k] = -smp.mat[k_fila][k];
	}
	
	smp.mat[k_fila][nc] = smp.x_sup[ix] - smp.mat[k_fila][nc];

	if (abs(smp.flg_x[ix]) <> 1) {
		writeln('mmmm ... porqué?');
	}
	smp.flg_x[ix] := -smp.flg_x[ix];
  
}


int pasoBuscarFactible(TDAOfSimplexGPUs &smp, int &cnt_RestrInfactibles) {
	
	int pFilaOpt, ppiv, qpiv, ix, res;
	double rval;
	bool filaFantasma, colFantasma;

	pFilaOpt = smp.NRestricciones - cnt_RestrInfactibles; // MAP: Antes nf - cnt_RestrInfactibles, indice ajustado
	// rval:= e(pFilaOpt, nc);
	rval = smp.mat[pFilaOpt][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice

	/* OJO LE AGREGO ESTE CHEQUEO PARA PROBAR */
	// Si parece satisfecha verifico que no se esté violándo la  fantasma
	if (rval > 0) {
		if (smp.left[pFilaOpt] < 0) {
			ix = -smp.left[pFilaOpt] - 1; // MAP: Ajusto indice con -1 
			if (smp.flg_x[ix] != 0) and (rval > smp.x_sup[ix]) {
				if (rval > smp.x_sup[ix] + CasiCero_Simplex) {
					cambiar_borde_de_caja(pFilaOpt);
					// rval:= e(pFilaOpt, nc );
					rval = smp.mat[pFilaOpt][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				} else {
					// pon_e(pFilaOpt, nc, x_sup.pv[ix]);
					smp.mat[pFilaOpt][smp.NVariables] = smp.x_sup[ix]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
					rval = smp.x_sup[ix];
				}
			}
		}
	} else if (rval > -CasiCero_Simplex) {
		// pon_e(pFilaOpt, nc, 0);
		smp.mat[pFilaOpt][smp.NVariables] = 0; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		rval = 0;
	}

	if (rval >= 0) {
		// ya es factible, probablemente se arregló con algún cambio anterior.
		cnt_RestrInfactibles--;
		res = 1;
	} else {
		// Nos planteamos el problema de optimización con objetivo el valor de la restricción violada
		if (cnt_RestrInfactibles > 0) {
			qpiv = locate_zpos(smp, pFilaOpt);
			if (qpiv > 0) {
				ppiv = mejorpivote(smp, qpiv, pFilaOpt, filaFantasma, colFantasma, true);
				if (ppiv < 1) {
					res = -1; // ShowMessage('No encontre pivote bueno ');
				} else {
					if (!colFantasma) {
						intercambiar(ppiv, qpiv);
						if (filaFantasma) {
							cambio_var_cota_sup_en_columna(smp, qpiv);
						}
						// if ( e( pFilaOpt, nc) >= 0 ) then
						if (smp.mat[pFilaOpt][smp.NVariables] >= 0) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
							cnt_RestrInfactibles--;
						}
						res = 1;
					} else {
						cambio_var_cota_sup_en_columna(smp, ppiv);
						// if ( e( pFilaOpt, nc) >= 0 ) then
						if (smp.mat[pFilaOpt][smp.NVariables] >= 0) { // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
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

		if (res = -1) {
			// Pruebo si soluciono la infactibildad con un intercambio de la infactible con una de las Activas
			qpiv = locate_qOK(pFilaOpt, smp.NVariables - cnt_columnasFijadas - 1, smp.NVariables); // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			if (qpiv > 0) {
				intercambiar(smp, pFilaOpt, qpiv);
				cnt_RestrInfactibles--;
				res = 1;
			}
		}
	}

	return res;
}

// Revisa si puede decrementar la cantidad de restricciones infactibles y lo hace en caso de poder hacerlo
// MAP: Eso de revisa se ve que me lo debe, declaro en el nombre del padre de hijo y del espiritu santo este procedimeinto como procedimiento al p... y sustituyo las llamadas por cnt_RestrInfactibles--
void decCnt_RestrInfactibles(int &cnt_RestrInfactibles) {
	cnt_RestrInfactibles--;
}

// Buscamos la columna que en la ultima fila (fila z) tenga el valor positivo mas grande retorna el número de columna si lo encontro, -1 si son todos < 0
// Este paso se da en el Simplex para minimizar, en el de maximizar busca el menos negativo
int locate_zpos(TDAOfSimplexGPUs &smp, int kfila_z) {
	int j, ires;
	double maxval;
	ires = -1;
	maxval = CasiCero_Simplex;
	for (j = 0; j <= smp.NVariables - cnt_columnasFijadas; j++) { // MAP: Antes 1 to nc - 1 - cnt_columnasFijadas
		if (smp.mat[kfila_z][j] > maxval) {
			maxval = smp.ma[tkfila_z][j];
			ires = j;
		}
	}
	return ires;
}


// MAP: Este procedimeinto es interno al procedimiento mejorpivote. ESTE PROCEDIMIENTO HABRIA QUE ELIMINARLO Y PEGAR EL CODIGO DIRECTO EN mejorpivote
void capturarElMejor(int i, double &a_iq_DelMejor, double &a_it_DelMejor, int &p, bool &filaFantasma, bool &colFantasma, double a_iq, double a_it, int i, bool xfantasma_Fila) {
  a_iq_DelMejor = a_iq;
  a_it_DelMejor = a_it;
  p = i;
  filaFantasma = xfantasma_Fila;
  colFantasma = false; // Solo por si el primero era la Fantasma de la Columna
}


int mejorpivote(TDAOfSimplexGPUs &smp, int q, int kmax, bool &filaFantasma, bool &colFantasma, bool checkearFilaOpt) {
  
	int i, p, ix;
	double a_iq, a_it, abs_a_pq,
		a_iq_DelMejor, a_it_DelMejor, abs_a_pq_DelMejor;
	bool xfantasma_Fila, esCandidato;

	// inicializaciones no necesarias, solo para evitar el warning
	a_it = 0;
	xfantasma_Fila = false;
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
		writeln( 'OPATROPA!');
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

	for (i = cnt_RestriccionesRedundantes; i <= kmax - 1, i++) { // MAP: Antes cnt_RestriccionesRedundantes + 1 to kmax - 1, muevo indice para ajustar a la nueva indexacion desde 0, kmax ya viene indexado con el ejuste
		// b(i) >= 0 para todo i / cnt_RestriccionesRedundantes < i < kmax-1
		// Buscamos la fila i que tenga el maximo b(i)/a(i,q) con a(i,q) < 0
		// aiq:= e( i, q );
		a_iq = smp.mat[i][q];
		if (a_iq >  CasiCero_Simplex) { // Si es positivo, verificamos si se trata de una x y entonces agregamos la fantasma.
			ix = -smp.left[i];
			if ((ix > 0) && (abs(smp.flg_x[ix - 1]) ==1)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
				// La variable en la fila i tiene cota superior, hay que probar con el cambio de variable
				a_iq = -a_iq;
				a_it = smp.x_sup[ix] - smp.mat[i][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				xfantasma_Fila = true;
				esCandidato = true;
			} else {
				esCandidato = false;
			}
		} else if (a_iq < -CasiCero_Simplex) { // Si es negativo es candidato
			a_it = smp.mat[i][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			esCandidato = true;
			xfantasma_Fila = false;
		} else {
			esCandidato = false;
		}


		if (esCandidato) { // Considero el coeficiente para elegir el pivote
			abs_a_pq = abs(smp.mat[i][q])); // MAP: Cambio abs( e( i, q ) ); por abs(smp.mat[i][q]));
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
				capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
				
			} else if (( a_it_DelMejor * a_iq) < ( a_it * a_iq_DelMejor ))  {
				capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
				
			} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
				if (abs_a_pq > abs_a_pq_DelMejor) {
					capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
				}
			}
		}
	}
	
	
	//MEJORA de codigo apturarElMejor( i ) es la misma instruccion en todos los casos del if elsif por lo que puede ser englobado en una sola condicion,  i:= kmax; no le veo objeto a esta definicion se puede usar kmax directamente y es mas claro el codigo
	if (checkearFilaOpt) {
		i = kmax;
		ix = -left[kmax];
		if (ix > 0) {  // Es una fila "x"
			if ( abs(flg_x[ix] ) = 1) {   // tiene manejo de cota superior
				begin
				// agregamos su fila fantasma como una más candidata a pivotear y a controlar
				// su factibilidad en caso de pivotear con otra.
				// En la fila kmax el aiq es positivo, pues fue elegido con locate_zpos
				//      aiq:= -e(kmax, q);
				a_iq = -smp.mat[kmax][q];
				a_it = smp.x_sup[ix - 1] - smp.mat[kmax][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
				abs_a_pq = abs(a_iq);
				assert(a_iq < 0, 'aiq >= 0 en tsimplex.mejorpivote');
				//      b_:= x_sup.pv[ix] - e(kmax, nc);
				xfantasma_Fila = true;
				if (p < 0) { // es el primer candidato
					capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
				} else if (( a_it * a_iq_DelMejor ) > ( a_it_DelMejor * a_iq)) {
					capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
				} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
					if (abs_a_pq > abs_a_pq_DelMejor) {
						capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
					}
				}
			}
		}


		// ahora bien, independientemente de que se trate de una x o una y
		// a( kmax, q ) > 0 por selección del q y a( kmax, ti ) < 0 porque
		// por eso estamos tratando de optimizar chequeando kmax para volverla
		// factible.
		// Agregamos entonces la posibilidad de pivotear con kmax como forma de volverla factible.
		a_iq = smp.mat[kmax][q];
		a_it = smp.mat[kmax][smp.NVariables]; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
		xFantasma_fila = false;
		abs_a_pq = abs(a_iq);
		if (p < 0) { // Es el primer candidato
			capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
		} else if (( a_it * a_iq_DelMejor ) < ( a_it_DelMejor * a_iq)) { // OJO, observar que es un "<"
			capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
		} else if (( a_it * a_iq_DelMejor ) == ( a_it_DelMejor * a_iq)) {
			if (abs_a_pq > abs_a_pq_DelMejor) {
				capturarElMejor(i, a_iq_DelMejor, a_it_DelMejor, p, filaFantasma, colFantasma, a_iq, a_it, i, xfantasma_Fila);
			}
		}
	}

	return p;
}


bool cambio_var_cota_sup_en_columna(TDAOfSimplexGPUs &smp, int q) {

	int ix, kfil;
	double xsup;
	bool res;
  
	res = false;
	ix = -top[q];
	// MAP: Agrego -1 para correr el indice a la indexacion desde 0 
	if ((ix > 0) && (abs(smp.flg_x[ix - 1]) == 1)) { // Corresponde a una x con cota sup
		// if abs(flg_x[ix] ) <> 1  then  writeln( 'mmmmm ' );

		// cambio de variable en la misma columna
		smp.flg_x[ix - 1] = -smp.flg_x[ix - 1]; // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
		xsup = smp.x_sup[ix - 1]; // MAP: Agrego -1 para correr el indice a la indexacion desde 0
		
		// MAP: Aca el codigo original esta dividido en 3 casos segun las restricciones redundantes y la variable de compilacion MODIFICAR_REDUNDANTES, como nosotros consideramos la variable siempre activa concateno todo en un caso con un unico for
		for (kfil = 0; kfil <= smp.NRestricciones; kfil++) { // MAP: Antes 1 to cnt_RestriccionesRedundantes
			smp.mat[kfil][smp.NVariables] = smp.mat[kfil][smp.NVariables] + smp.mat[kfil][q] * xsup; // MAP: Cambio nc por smp.NVariables, dado que nc = smp.NVariables - 1 entonce ajusta el indice
			smp.mat[kfil][q] = -smp.mat[kfil][q];
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



int locate_qOK(TDAOfSimplexGPUs &smp, int p, int jhasta, int jti) {
	int mejorq, q;
	double max_apq, apq;

	mejorq = -1;
	max_apq = -1;
	for (q = 0; q <= jhasta; q++) { // MAP: Antes 1 to jhasta, jhasta ya viene con el indice ajustado a la indexacion por 0
		if (test_qOK(smp, p, q, jti, apq) && ((mejorq < 0) or (apq > max_apq))) {
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
bool test_qOK(TDAOfSimplexGPUs &smp, int p, int q, int jti, double &apq) {
	int k, ix;
	double alfa_p, akq,
		nuevo_ti;

	// apq:= e(p, q);
	apq = smp.mat[p][q];
	// if ( apq  <= AsumaCero) then
	if (abs(apq) <= AsumaCero) { // rch@202012081043 agrego el abs()
		return false;
	} else {
		// alfa_p:= -e( p, jti ) / apq;
		alfa_p = -smp.mat[p][jti] / apq;
		ix = -top[q];
		if (ix > 0) { // la col q es una x
			if (smp.flg_x[ix - 1] != 0) {  // tiene manejo de cotasup, MAP: Agrego -1 para correr el indice a la indexacion desde 0 
				if (alfa_p > smp.x_sup[ix - 1]) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
					// De intercambiar esta columna se violaría la cotas superior
					if (alfa_p > (smp.x_sup.pv[ix - 1] + CasiCero_Simplex)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
						return false;
					}
					// else begin //writeln( 'OJO, TSimplex.test_qOk, de intercambiar esta  columna se viola por poquito la cota sup y la dejo pasar' );
				}
			}
		}
		
		for (k = cnt_RestriccionesRedundantes; k < p; k++) { // MAP: cnt_RestriccionesRedundantes + 1 to p - 1, p ya viene con el indice ajustado a la indexacion desde 0
			// akq = e(k, q);
			akq = smp.mat[k][q];
			// nuevo_ti = e(k,jti) + akq * alfa_p;
			nuevo_ti = smp.mat[k][jti] + akq * alfa_p;
			if (nuevo_ti < -CasiCero_Simplex) {
				return false
			} else {
				ix = -left[k];
				if ((ix > 0) and (smp.flg_x[ix - 1] != 0)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
					//  if nuevo_ti > x_sup.pv[ix]  then
					if (nuevo_ti > (smp.x_sup[ix - 1] + CasiCero_Simplex)) { // MAP: Agrego -1 para correr el indice a la indexacion desde 0 
						return false
					}
				}
			}
		}

		return true;
	}
}



int darpaso(TDAOfSimplexGPUs &smp) {

	int ppiv, qpiv,
		res;
	bool filaFantasma, colFantasma;

	cnt_paso++;
	qpiv = locate_zpos(smp.NRestricciones); // MAP: Cambio nf por smp.NRestricciones, dado que nf = smp.NRestricciones - 1 entonce ajusta el indice
	if (qpiv > 0) {
		ppiv = mejorpivote(qpiv, nf, filaFantasma, colFantasma, false);
		if (ppiv < 1) {
			return -1;
		}
		
		if (!colFantasma) {
			if  (!intercambiar(ppiv, qpiv)) {
				return -1;
			}
			
			if (filaFantasma) {
				if (!cambio_var_cota_sup_en_columna(qpiv) {
					return -1;
				}
			}
			res = 1;
		} else {
			assert(ppiv = qpiv , 'Si Es FantasmaDeCol tenía que ser ppiv = qpiv ');
			cambio_var_cota_sup_en_columna(ppiv);
			res = 1;
		} 
	} else {
		res = 0; // ShowMessage('No encontre z - positivo ' );
	}
		
	return res;
}


/* MAP: AHORA DEBEMOS IMPLEMENTAR EL EJEMPLO EN C++ PARA ARRANCAR LAS PRUEBAS CON DATOS FACILES DE CHEQUEAR, 
LAS FUNCIONES QUE NECESITA ESTE EJEMPLO DEBEN SER TRAIDAS DEL ORIGINAL E IMPLEMENTADAS TAMBIEN

procedure ejemplo;
var
  i: integer;
  spx: TSimplex;
begin
  {
  min z = x1 + 3x2 + 2x3
  s.a.
    x1 + x2 + x3 >= 10.5
    x1 + x2 = 5.3
    x1 - x3 <= 2.9
    0 <= x1 <= 12, -6 <= x2 <= 6, -5 <= x3 <= 5

  =>

  max -z = -x1 -3x2 -2x3
  s.a.
    x1 + x2 + x3 -10.5 >= 0
    x1 + x2 - 5.3 = 0
    -x1 + x3 + 2.9 >= 0
    0 <= x1 <= 12, -6 <= x2 <= 6, -5 <= x3 <= 5
  }

  //Creamos un simplex vacío cuya matriz M tendrá:
  //3 restricciones + la función objetivo
  //3 variables + los términos independientes
  spx := TSimplex.Create_init(4, 4, nil, nil);

  //Cargamos la fila 1, pon_e(k, j, x) hace Mkj:= x
  spx.pon_e(1, 1, 1);
  spx.pon_e(1, 2, 1);
  spx.pon_e(1, 3, 1);
  spx.pon_e(1, spx.nc, -10.5);

  //Cargamos la fila 2 y la declaramos como de igualdad
  spx.pon_e(2, 1, 1);
  spx.pon_e(2, 2, 1);
  spx.pon_e(2, 3, 0);
  spx.pon_e(2, spx.nc, -5.3);
  spx.FijarRestriccionIgualdad(2);

  //Cargamos la fila 3
  spx.pon_e(3, 1, -1);
  spx.pon_e(3, 2, 0);
  spx.pon_e(3, 3, 1);
  spx.pon_e(3, spx.nc, 2.9);

  //Cargamos la fila objetivo z
  spx.pon_e(spx.nf, 1, -1);
  spx.pon_e(spx.nf, 2, -3);
  spx.pon_e(spx.nf, 3, -2);

  //cota_inf_set(i, x) fija la cota inferior de la variable en la
  //posición i a x, sota_sup_set hace lo propio con la cota superior
  //Cotas inferior y superior de x1
  spx.cota_inf_set(1, 0);
  spx.cota_sup_set(1, 12);

  //Cotas inferior y superior de x2
  spx.cota_inf_set(2, -6);
  spx.cota_sup_set(2, 6);

  //Cotas inferior y superior de x3
  spx.cota_inf_set(3, -5);
  spx.cota_sup_set(3, 5);

  //Vuelco el simplex al archivo 'ProblemaEjemplo.xlt' para verificar
  //que el problema armado sea el que quería
  //MAP COMENTED no needed now spx.DumpSistemaToXLT_('ProblemaEjemplo.xlt', '');

  //intento resolver
  if spx.resolver = 0 then
  begin
    //ok, encontró solución
    Writeln('Solución óptima encontrada:');
    //spx.fval obtiene el valor de z
    Writeln('z= ', FloatToStrF(-spx.fval, ffGeneral, 8, 4));
    Writeln;
    for i := 1 to 3 do
      //spx.xval(i) obtiene el valor de la variable i
      Writeln(#9, spx.fGetNombreVar(i), '= ', FloatToStrF(spx.xval(i), ffGeneral, 8, 3));
    Writeln;
    for i := 1 to 3 do
      //spx.yval(i) obtiene el valor de la restriccion i
      Writeln(#9, spx.fGetNombreRes(i), '= ', FloatToStrF(spx.yval(i), ffGeneral, 8, 3));
    Writeln('Presione <Enter> para continuar');
    Readln;
  end
  else
    //Error, lanzamos la excepción
    raise Exception.Create('Error resolviendo simplex: ' + spx.mensajeDeError);

  //Liberamos la memoria usada por el objeto
  spx.Free;
end;

*/