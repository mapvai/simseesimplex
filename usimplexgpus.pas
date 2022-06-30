unit usimplexgpus;

{$mode delphi}

{$linklib c}
{$linklib libgcc_s.so.1}
{$linklib libstdc++.so.6}
{$linklib libm.so}
{$linklib libcuda.so}
{$link /usr/local/cuda/lib64/libcudart.so}

{$link cuda/ini_mem.o}
//{$link cuda/resolver_cuda_gpu.o}
//{$link cuda/resolver_our_s_cpu.o}
//{$link cuda/resolver_simplex_big_m.o}
//{$link cuda/resolver_simplex_big_m2.o}
{$link cuda/resolver_simplex_big_m_final.o}
{$link cuda/free_mem.o}

interface
  uses
    xmatdefs;

  type

    TSimplexGPUs = array of NReal;

    TSimplexGPUsOLD = record
      NEnteras,NVariables,NRestricciones,cnt_varfijas,
      cnt_RestriccionesRedundantes:Integer;
      x_inf,x_sup:^NReal; // largo NVariables  todo indexado de 0 a N - 1
      flg_x : ^ShortInt; // largo NVariables
      top : ^Integer;
      flg_y : ^ShortInt; // largo NRestricciones
      left : ^Integer;
      lstvents: ^Integer; // largo NEnteras
      lstAcoplesVEnts: TDAOfShortInt; // FLATTEN largo NEnteras*2, asumiendo que casa vEnt tiene 1 solo acople
      mat: TDAOfNReal; // FLATTEN filas = NRestricciones + 1 (por la funcion a maximizar
                           // columnas = NVariables + 1 (por el termino independiente)
    end;

    TDAOfSimplexGPUs = array of TSimplexGPUs;

    PTSimplexGPUs = ^TSimplexGPUs;


    TSimplexGPUsOLD = record
      NEnteras,NVariables,NRestricciones,cnt_varfijas,
      cnt_RestriccionesRedundantes:Integer;
      x_inf,x_sup:^NReal; // largo NVariables  todo indexado de 0 a N - 1
      flg_x : ^ShortInt; // largo NVariables
      top : ^Integer;
      flg_y : ^ShortInt; // largo NRestricciones
      left : ^Integer;
      lstvents: ^Integer; // largo NEnteras
      lstAcoplesVEnts: TDAOfShortInt; // FLATTEN largo NEnteras*2, asumiendo que casa vEnt tiene 1 solo acople
      mat: TDAOfNReal; // FLATTEN filas = NRestricciones + 1 (por la funcion a maximizar
                           // columnas = NVariables + 1 (por el termino independiente)
    end;

    TOurSimplexGPUs = record
      NEnteras,NVariables,NRestricciones,cnt_varfijas,
      cnt_RestriccionesRedundantes:Integer;
      x_inf,x_sup:^NReal; // largo NVariables  todo indexado de 0 a N - 1
      flg_x : ^ShortInt; // largo NVariables
      top : ^Integer;
      flg_y : ^ShortInt; // largo NRestricciones
      left : ^Integer;
      lstvents: ^Integer; // largo NEnteras
      lstAcoplesVEnts: TDAOfShortInt; // FLATTEN largo NEnteras*2, asumiendo que casa vEnt tiene 1 solo acople
      mat: TDAOfNReal; // FLATTEN filas = NRestricciones + 1 (por la funcion a maximizar
                           // columnas = NVariables + 1 (por el termino independiente)
    end;

    TDAOfOurSimplexGPUs = array of TOurSimplexGPUs;

    PTOurSimplexGPUs = ^TOurSimplexGPUs;

  procedure ini_mem(var d_simplex_array : PTSimplexGPUs;
                    var h_simplex_array : TDAOfSimplexGPUs;
                    NTrayectorias : Int32;
                    NEnteras : Int32;
                    NVariables : Int32;
                    NRestricciones : Int32;
                    cnt_varfijas : Int32;
                    cnt_RestriccionesRedundantes : Int32);cdecl;external;

  procedure resolver_cuda(var simplex_array : TDAOfSimplexGPUs;
                          var d_simplex_array : PTSimplexGPUs;
                          var h_simplex_array : TDAOfSimplexGPUs;
                          NTrayectorias : Int32) ;cdecl;external;

  procedure free_mem(var d_simplex_array : PTSimplexGPUs;
                    var h_simplex_array : TDAOfSimplexGPUs;
                    NTrayectorias : Int32;
                    NEnteras : Int32;
                    NVariables : Int32;
                    NRestricciones : Int32;
                    cnt_varfijas : Int32;
                    cnt_RestriccionesRedundantes : Int32);cdecl;external;

implementation

end.

