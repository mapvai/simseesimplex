struct TSimplexGPUs {
    int NVariables, NRestricciones, NColumnas, Nfilas, cantArtVars;
    double* tabloide; // filas = NRestricciones + 1 (por la funcion a maximizar), columnas = NVariables + 1 (por el termino independiente)
};

typedef TSimplexGPUs* TDAOfSimplexGPUs;
