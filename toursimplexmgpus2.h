struct TSimplexGPUs {
    int filas, columnas;
    int* top; // largo filas
    int* left; // largo columnas
	int* var_type; // largo filas
	double* sup; // cantidad de variables 
	double* inf; // cantidad de variables 
	double* Cb; // largo filas
    double* tabloide; // filas = NRestricciones + 1 (por la funcion a maximizar), columnas = NVariables + 1 (por el termino independiente)
};

typedef TSimplexGPUs* TDAOfSimplexGPUs;
