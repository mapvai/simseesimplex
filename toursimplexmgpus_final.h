typedef double* TabloideGPUs;

typedef TabloideGPUs* TDAOfSimplexGPUs;

// internal
struct TSimplexGPUs {
    int filas, columnas;
	int var_x, var_s, var_a, var_all;
	int rest_ini, rest_fin;
	int mat_adv_row;
	
	double* tabloide; // Referencia al tabloide
	
    double* top; // largo filas, horizontal
    double* left; // largo columnas, vertical
	
	double* var_type; // largo filas, horizontal
	
	double* flg_x; // 0 NO cota superior, 1 cota superior, 2 variable fijada, horizontal
	
	double* flg_y; // 0 restriccion >=, 1 <=, 2 =, vertical
	
	double* sup; // cantidad de variables, horizontal
	double* inf; // cantidad de variables, horizontal
	double* Cb; // cantidad de restricciones, vertical

	double* Xb; // cantidad de restricciones, vertical
	double* z; // funcion z, cantidad de variables, horizontal
    double* matriz; // restricciones x variables con avance fila largo de tabloide (mat_adv_row)
	double* mat_ext; // Matriz extendida Xb|A
};
