struct TSimplexGPUs {
    int NEnteras,
        NVariables,
        NRestricciones,
        cnt_varfijas,
        cnt_RestriccionesRedundantes;
    double* x_inf;
    double* x_sup; // largo NVariables  todo indexado de 0 a N - 1
    int8_t* flg_x;
    int* top; // largo NVariables
    int8_t* flg_y;
    int* left; // largo NRestricciones
    int* lstvents;
	char* lstAcoplesVEnts;
    double* mat; // filas = NRestricciones + 1 (por la funcion a maximizar
                         // columnas = NVariables + 1 (por el termino independiente)
};

typedef TSimplexGPUs* TDAOfSimplexGPUs;


