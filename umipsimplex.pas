{xDEFINE DBG_CNT_MIPSPX}

{xDEFINE NORMALIZAR_RESTRICCIONES}
{xDEFINE ABORTARAOPTIMIZACIONSINOESMEJORABLE}
//Corta una rama si el (fVal - fValMejor) <= ( fValRoot - fValMejor ) * fValMargen
// fValRoot es el valor obtenido al resolver el nodo raiz (es el máximo valor obtenible (TODO RELAJADO y estamos MAXIMIZANDO).


{xDEFINE RELAJACIONENCADENA}
//Intenta desrelajar primero variables que en el padre estuvieran relajadas

{xDEFINE DBG_CONTAR_CNT_MIPSIMPLEX}
//Lleva un contador de la cantidad de nodos que le quedan por resolver
//en el arbol actual

{xDEFINE MIP_DBG}

unit umipsimplex;
{$MODE Delphi}
// {$mode objfpc}{$H+}

interface

uses
  SysUtils, xMatDefs, Math, usimplex, MatReal,
  {$IFDEF SIMSEE_GPUs}
  usimplexgpus,
  {$ENDIF}
  {$IFDEF TRATAR_GLPK}
  glpk,
  {$ENDIF}
  {$IFDEF VIOLACIONES_PERMITIDAS}
  uListaViolacionesPermitidasSimplex,
  {$ENDIF}
  classes,
  uauxiliares, MatEnt;

{$IFDEF ABORTARAOPTIMIZACIONSINOESMEJORABLE}
const
  fvalMargen = 1.0/1000.0;//Si el ( fval - fValMejor ) <= (fValMejor -fValRoot ) * fvalMargen  no sigo pues la máxima mejora es muy chica
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
const
  cantNodosSignificativa = 1.0E+18;
{$ENDIF}

type
  TEstadoFichaNodo = (
    CN_NoAsignado,
    CN_EliminadoPor_fval,
    CN_EliminadoPor_infactible,
    CN_MejorFactible,
    CN_Relajado);


  ContadorParaCombinatoria = system.extended;
  TFichaNodoProblema = class;
  TDAOfFichaNodoProblema = array of TFichaNodoProblema;
  TDAOfAcoplesVEnts = array of TListaAcoplesVEntera;

  { TFichaNodoProblema }

  TFichaNodoProblema = class
  public
    {$IFDEF MIPSIMPLEXFROMORIGINAL}
    spxOriginal: TSimplex;
    {$ENDIF}
    Padre: TFichaNodoProblema;
    esFactible: boolean;
    Estado: TEstadoFichaNodo;
    kx_Relajada: integer; // indice de la variable entera relajada en base a la cual
    // se generan los sub-arboles.
    x: TDAofNReal; // resultado del simplex
    //y, xmult, ymult: TDAOfNReal; // resultado del simplex
    fval: NReal; // valor de la función a maximiar para esta sol.
    nodo_izq: TFichaNodoProblema;
    nodo_der: TFichaNodoProblema;
    spx: TSimplex; // guarda una copia del problema con las modificaciones
    // de cotas que identifican al nodo. spx no se usa para
    // resolver, se usa sólo para crear las ramas del nodo
    // agregando las modificaciones de cota de cada rama

    res_spx: integer; // resultado del resolver del simplex

    arbolCerrado: boolean;
    lstvents: TDAOfNInt;
    lstAcoplesVEnts: TDAOfAcoplesVEnts;
{$IFDEF RELAJACIONENCADENA}
    relajadas: TDAOfBoolean;
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    nVarsNoFijadas: integer;
{$ENDIF}
    kDesrelajar: integer;
    ires: NReal; // valor de la variable relajada en el padre
    sentido: integer;

    constructor Create_Raiz(xspx  {$IFDEF MIPSIMPLEXFROMORIGINAL}, xspxOriginal{$ENDIF}: TSimplex;
    // pasa el problema relajado del padre sin resolver
      xlstvents: TDAOfNInt; // es la lísta de los indices de las variables enteras
      xlstAcoplesVEnts: TDAOfAcoplesVEnts; flgClone: boolean = true );

    constructor Create_Desrelajando(xPadre: TFichaNodoProblema;
      xspx  {$IFDEF MIPSIMPLEXFROMORIGINAL}, xspxOriginal{$ENDIF}: TSimplex; // pasa el problema relajado del padre sin resolver
      xlstvents: TDAOfNInt; // es la lísta de los indices de las variables enteras
      xlstAcoplesVEnts: TDAOfAcoplesVEnts; xkEnteraDesrelajar: integer;
    // es el indice en la lista de variables enteras que corresponde a la variable a desrelajar
      xires: NReal; // es el resultado de la optimización para la variable a desrelajar
      xsentido: integer // -1 es desrelajar hacia la izquierda, 1 hacia la derecha
      );
    destructor destroy; override;
    function Solve(ElNodoRaiz: TFichaNodoProblema;
      var ElMejorNodoFactible: TFichaNodoProblema;
      const Carpeta_ParaDump: string): boolean;
    procedure PrintSolEncabs(var f: textfile);
    procedure PrintSolVals(var f: Textfile);
    function primerCerradoEnLaCadenaHaciaArriba: TFichaNodoProblema;
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    function cnt_Relajaciones_Hacia_Abajo: ContadorParaCombinatoria;
    //Retorna la cantidad de nodos que se pueden relajar a partir de la ficha actual
{$ENDIF}
    function GetFichaRaiz: TFichaNodoProblema;
  end;

  { TMIPSimplex }

  TMIPSimplex = class(TSimplex)
  public
    ElNodoRaiz, ElMejorNodoFactible: TFichaNodoProblema;
    nvents: integer;
    //Se indexan desde 0
    lstvents: TDAOfNInt; // lista de las enteras.
    lstAcoplesVEnts: TDAOfAcoplesVEnts;

    constructor Create_init(mfilas, ncolumnas, nenteras: integer;
      xfGetNombreVar, xfGetNombreRes: TFuncNombre);
      reintroduce; virtual;

    constructor CreateFromXLT(var f: textfile); override;

    function resolver: integer; override;

    function SimplexSolucion: TSimplex;
    procedure set_entera(ivae, ivar: integer; CotaSup: integer); override;
    procedure set_EnteraConAcople(ivae, ivar: integer; CotaSup: integer;
      ivarAcoplada, iresAcoplada: integer); override;

    procedure set_EnteraConAcoples(ivae, ivar: integer; CotaSup: integer;
      lstAcoples: TListaAcoplesVEntera); override;


    procedure fijarCotasSupDeVariablesAcopladas;
    procedure limpiar; override;
    destructor destroy; override;

    procedure CopiarDefinicionesEnteras( MSPX: TMipSimplex );


   (*
     Funciones auxiliares para leer los resultados
   *)
    function xval(ix: integer): NReal; override;
    function yval(iy: integer): NReal; override;
    function xmult_caja(ix: integer; var aflg_x: shortint): NReal; override;
    function xmult(ix: integer): NReal; override;
    function ymult(iy: integer): NReal; override;
    function fval: NReal; override;

    procedure DumpSistemaToXLT(var f: textfile); override;

    procedure DumpSistemaToBIN( f: TStream ); override;
    constructor CreateFromBin( f: TStream ); override;

    // retorna TRUE si x_j es una variable entera
    function EsEntera( j: integer ): boolean; override;
    {$IFDEF SIMSEE_GPUS}
    procedure toStruct(var struct : TSimplexGPUs);
    procedure loadFromStruct(struct : TSimplexGPUsOLD);
    procedure loadFromOurStruct(struct : TSimplexGPUsOLD);
    {$ENDIF}


  private
    function trySolve: boolean;

  end;

  TDAOfMIPSimplex = array of TMIPSimplex;


(*+doc spxActivo
Esta variable la usamos para apuntar al Simplex que está bajo resolución.
Lo usamos para monitoreo del simplex. Cuando no está en rsolución un simplex,
esta variable está a nil -doc*)
var
  spxActivo: TSimplex;
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
  nodos_Totales, nodos_Recorridos: ContadorParaCombinatoria;
  acum_Nodos_Recorridos: ContadorParaCombinatoria;//acumula la cantidad de nodos
//recorridos hasta que sea un
//valor significativo
{$ENDIF}

implementation
{$IFDEF TRATAR_GLPK}
uses
   umipsimplex_glpk;

{$IFDEF DBG_CNT_MIPSPX}
var
  CNT_MIPSPX: integer;
{$ENDIF}

function Tratar_GLPK( MipSPX: TMIPSimplex ): integer;
var
  MipSPX_glpk: TMIPSimplex_glpk;
  res: integer;
  {$IFDEF SPXCONLOG}
  s: string;
  k: integer;
  {$ENDIF}

procedure dump_write_sol;
var
  f: textfile;
  k: integer;
begin
  assignfile(f, 'resultado_tratar_glpk.xlt' );
rewrite( f );
writeln(f, 'fobj: '+ FloatToStr( MipSPX_glpk.fval ) );
for k:= 1 to MipSPX_glpk.nc-1 do
  writeln( f, FloatToStr( MipSPX_glpk.xval( k ) ) );
closefile( f );
end;

begin
//  MipSPX.DumpSistemaToXLT_( 'original_preGLPK.xlt', 'original_preNormalizar');
  MipSPX_glpk:= TMIPSimplex_glpk.CreateFromMipSimlex( MipSPX );

  // MipSPX_glpk.DumpSistemaToXLT_('dumpglpk.lxt', 'dumpglpk' );
  // MipSPX_glpk.DumpSistemaToOctaveFile('dumpOctable_glpk.m', '' );
  {$IFDEF SPXCONLOG}
  MipSPX.writelog('Falló el MIPSimplex clásico, pruebo con GLPK.' );
  {$ENDIF}
  res:= MipSPX_glpk.resolver;
  if  res < 0 then
  begin // no tuvo suerte
    result:= -1;
    {$IFDEF SPXCONLOG}
    MipSPX.writelog('Falló también GLPK, el problema debe ser realmente INFACTIBLE.' );
    {$ENDIF}
    exit;
  end;

  {$IFDEF SPXCONLOG}
  MipSPX.writelog('GLPK tuvo ÉXITO!!!.' );
  MipSPX.writelog('fobj: '+ FloatToStr( MipSPX_glpk.fval ) );
  s:= 'Sol: '+FloatToStr( MipSPX_glpk.xval( 1 ) );
  for k:= 2 to MipSPX_glpk.nc-1 do
    s:= s+', '+FloatToStr( MipSPX_glpk.xval( k ) );
  MipSPX.writelog( s );
  {$ENDIF}

//  dump_write_sol;

  if MipSPX.ElNodoRaiz <> nil then
    MipSPX.ElNodoRaiz.Free;
  MipSPX.ElNodoRaiz := TFichaNodoProblema.Create_Raiz( MipSPX_glpk, {$IFDEF MIPSIMPLEXFROMORIGINAL}nil,{$ENDIF} nil,nil, false );
  MipSPX.ElMejorNodoFactible:= MipSPX.ElNodoRaiz;
  result:= res;
end;

{$ENDIF}


{$IFDEF TRATAR_NORMALIZAR_SPX}
function Tratar_Normalizar_SPX( MipSPX: TMIPSimplex ): integer;
var
  aMSPX: TMIPSimplex;
  res: boolean;
  {$IFDEF SPXCONLOG}
  s: string;
  k: integer;
  {$ENDIF}
begin
//  MipSPX.DumpSistemaToXLT_( 'original_preNormalizar.xlt', 'original_preNormalizar');
  aMSPX:= TMIPSimplex.Create_clone( MipSPX );
  aMSPX.CopiarDefinicionesEnteras( MipSPX );
  aMSPX.Normalizar;

//  aMSPX.DumpSistemaToXLT_( 'Normalizado.xlt', 'Normalizado');
  {$IFDEF SPXCONLOG}
  aMSPX.writelog('Falló el MIPSimplex clásico, pruebo con normalizarlo.' );
  {$ENDIF}
  res:= aMSPX.TrySolve;
  if  not res then
  begin // no tuvo suerte
    result:= -1;
    {$IFDEF SPXCONLOG}
    aMSPX.writelog('Falló también la solución normalizada, el problema debe ser realmente INFACTIBLE.' );
    {$ENDIF}
    exit;
  end;


  {$IFDEF SPXCONLOG}
  aMSPX.writelog('Normalizado tuvo ÉXITO!!!.' );
  aMSPX.writelog('fobj: '+ FloatToStr( aMSPX.fval ) );
  s:= 'Sol: '+FloatToStr( aMSPX.xval( 1 ) );
  for k:= 2 to aMSPX.nc-1 do
    s:= s+', '+FloatToStr( aMSPX.xval( k ) );
  aMSPX.writelog( s );
  {$ENDIF}

  if MipSPX.ElNodoRaiz <> nil then
    MipSPX.ElNodoRaiz.Free;
  MipSPX.ElNodoRaiz := TFichaNodoProblema.Create_Raiz( aMSPX  {$IFDEF MIPSIMPLEXFROMORIGINAL}, nil{$ENDIF}, nil,nil, false );
  MipSPX.ElMejorNodoFactible:= MipSPX.ElNodoRaiz;
  result:= 0;
end;

{$ENDIF}


constructor TMIPSimplex.Create_init(mfilas, ncolumnas, nenteras: integer;
  xfGetNombreVar, xfGetNombreRes: TFuncNombre);
var
  i: integer;
begin
  inherited Create_init(mfilas, ncolumnas, xfGetNombreVar, xfGetNombreRes);
  setlength(lstvents, nenteras);
  setLength(lstAcoplesVEnts, nenteras);
  nvents := nenteras;
  for i := 0 to nenteras - 1 do
  begin
    setLength(lstAcoplesVEnts[i], 1);
    lstAcoplesVEnts[i][0].ivar := -1;
    lstAcoplesVEnts[i][0].ires := -1;
  end;
  ElNodoRaiz := nil;
  ElMejorNodoFactible := nil;
end;

constructor TMIPSimplex.CreateFromXLT(var f: textfile);
var
  kvar: integer;
  k, j: integer;
  {$IFDEF VIOLACIONES_PERMITIDAS}
  ficha: TFichaViolacionPermitida;
  cnt: integer;
  {$ENDIF}
  aval: NReal;

{$IFNDEF SPXCONLOG}
//  nom: string;
{$ENDIF}

  NVariablesMas1, NRestriccionesMas1: integer;
  {%H-}pal: string;
  r: string;

function readCampo( nombre: string ): string;
var
  ipos: integer;
begin
  ipos:= pos( nombre, r );
  while ( ipos = 0 ) and not eof( f ) do
  begin
    readln( f, r );
    ipos:= pos( nombre, r );
  end;
  if ipos > 0 then
  begin
    delete( r, 1, ipos + length( nombre ) -1 );
    r:= trim( r );
    result:= nextpal(  r );
  end
  else
   raise Exception.Create( 'No encontré campo: '+ nombre );
end;

function readCampoInt( nombre: string ): integer;
var
  pal: string;
begin
  pal:= readCampo( nombre );
  result:= StrToInt( pal );
end;


var
  Separadores_BK: set of char;
  str: String;
  indicesEnteras: TVectE;
  kEntera, NEnteras: Integer;

begin
  Separadores_BK:= Separadores_Activos;
  Separadores_Activos:= [ #9 ];

  r:= '';

  readln(f,str); // Info adicional

  readln(f,str); // NEnteras: cntEnteras ...

  if pos( 'NEnteras:', str ) <> 0 then
  begin
    NextPal(str); // NEnteras:
    NEnteras:=NextInt(str);
    indicesEnteras:=TVectE.Create_Init(NEnteras);
    for kEntera:=1 to NEnteras do
      indicesEnteras.pon_e(kEntera,NextInt(str));
    NVariablesMas1:= readCampoInt('NVariables:' )+1;
  end
  else
  begin
    NEnteras:= 0;
    indicesEnteras:= TVectE.Create_Init( 0 );
    NextPal( str ); // NVariables:
    NVariablesMas1:= NextInt( str ) + 1;
  end;


  NRestriccionesMas1:= readCampoInt( 'NRestricciones:')+1;
  nvents:=0; // esto es para que el limpiar no recorra las listas
  Create_Init( NRestriccionesMas1, NVariablesMas1,NEnteras ,  nil, nil );

  cnt_VarFijas:= readCampoInt('cnt_varfijas:' );
  cnt_RestriccionesRedundantes:= readCampoInt('cnt_RestriccionesRedundantes:' );
  {$IFDEF VIOLACIONES_PERMITIDAS}
  cnt_violacionesUsadas:= readCampoInt('cnt_violacionesUsadas');
  readln(f ); // 'violacionesPermitidas');


  CNT:= readCampoInt( 'violacionesPermitidas.Count=' );
  readln(f, r );
//  writeln(f, 'ires', #9, 'usada', #9, 'iViolacionAUsar', #9, 'nIvars', #9, 'ivars[]');

  for k := 0 to cnt - 1 do
  begin
    ficha := TFichaViolacionPermitida.CreateFromXlt( f );
    violacionesPermitidas.Add( ficha );
  end;
  {$ENDIF}
  repeat      // con esto logro que si fue salvada con VIOLACIONES_PERMITIDAS lo lea igual
    readln(f, r ); //'*****************************');
  until pos( '******', r ) = 1;

  readln(f, r ); // Write(f, 'x: ');
  {$IFDEF SPXCONLOG}
  setlength( nombreVars, nc );
  setlength( nombreRest, nf );
  {$ENDIF}

  readln( f, r );  //  Write(f, 'x_inf:');
  pal:= nextpal( r );
  for kvar := 1 to nc - 1 do
     x_inf.pv[kvar]:= nextfloat( r );

  readln( f, r );
  pal:= nextpal( r ); //Write(f, 'x_sup:');
  for kvar := 1 to nc - 1 do
    x_sup.pv[kvar]:= nextfloat( r );

  readln( f, r );
  pal:= nextpal( r ); //Write(f, 'flg_x:');
  for kvar := 1 to nc - 1 do
    flg_x[kvar]:= nextint( r );

  cnt_igualdades:=0;
  readln( f, r );
  pal:= nextpal( r ); //Write(f, 'flg_y:');
  for kvar := 1 to nf - 1 do
  begin
    flg_y[kvar]:= nextint( r );
    if flg_y[kvar] = 2 then
      inc(cnt_igualdades);
  end;

  readln( f, r );
  pal:= nextpal( r );
  for kvar := 1 to nc - 1 do
    top[kvar]:= nextint( r );

  readln( f, r );
  pal:= nextpal( r ); //Write(f, 'left:');
  for kvar := 1 to nf - 1 do
    left[kvar]:= nextint( r );


  rearmarIndicesiiXiiY;

  readln( f, r ); // '----------------------');
  readln( f, r ); // 'sistema --------------');
  readln( f, r ); // '......................');

  // encabezados de las columnas
  readln(f , r );
  pal:= nextpal( r ); //Write(f, '-');
    {$IFDEF SPXCONLOG}
  for j := 1 to nc - 1 do
  begin
    pal:= nextpal( r );
    if top[j] < 0 then // es una x
      nombreVars[ -top[j] ]:= pal
    else
      nombreRest[ top[j] ]:= pal;
  end;
{$ENDIF}
  // filas ecuaciones >= 0
  for k := 1 to nf - 1 do
  begin
    readln( f, r );
    // writeln( r );

    pal:= nextpal( r );
    {$IFDEF SPXCONLOG}
    if left[k] > 0 then
      nombreRest[ left[k] ]:= pal
    else
      nombreVars[ -left[k] ]:= pal;
{$ENDIF}
    for j := 1 to nc do
    begin
      aval := nextfloat( r );
      pon_e(k, j, aval );
    end;

  for kEntera:=1 to nvents do
    set_entera(kEntera, indicesEnteras.e(kEntera),
      round(x_sup.e(indicesEnteras.e(kEntera))));

  (*
    if left[k] < 0 then
    begin
      Write(f, #9, '>= 0');
      if flg_x[-left[k]] <> 0 then
        Write(f, #9, ' <= ', FloatToStrF(x_sup.pv[-left[k]], ffFixed, 10, 3));
    end
    else if left[k] > 0 then
    begin
      if flg_y[k] <> 0 then
        Write(f, #9, '= 0')
      else
        Write(f, #9, '>=0');
      Write(f, #9);
    end
    else if flg_x[-left[k]] = 0 then
      Write(f, #9);

    writeln(f, #9, k);
    *)
  end;

  readln( f, r );
  // ultima fila (función a maximizar )
  pal:= nextpal( r ); //  Write(f, 'max:');
  for j := 1 to nc do
  begin
    aval:= nextfloat( r );
    pon_e(nf, j, aval );
  end;

  readln( f, r ); //writeln(f);
  readln( f, r ); // 1 2 3 4 ...

  Separadores_Activos:= Separadores_BK;

end;

procedure TMIPSimplex.limpiar;
var
  i: integer;
begin
  inherited limpiar;
  for i := 0 to nvents - 1 do
  begin
    setLength(lstAcoplesVEnts[i], 1);
    lstAcoplesVEnts[i][0].ivar := -1;
    lstAcoplesVEnts[i][0].ires := -1;
  end;
  if ElMejorNodoFactible <> nil then
  begin
    //    ElMejorNodoFactible.Free;
    ElMejorNodoFactible := nil;
  end;
  if ElNodoRaiz <> nil then
  begin
    ElNodoRaiz.Free;
    ElNodoRaiz := nil;
  end;
end;

destructor TMIPSimplex.destroy;
var
  iAcoples: integer;
begin
  if ElNodoraiz <> nil then
  begin
    ElNodoRaiz.Free;
    ElNodoRaiz := nil;
  end;
  ElMejorNodoFactible := nil;
  setlength(lstvents, 0);
  for iAcoples := 0 to high(lstAcoplesVEnts) do
    SetLength(lstAcoplesVEnts[iAcoples], 0);
  setLength(lstAcoplesVEnts, 0);
  inherited destroy;
end;

procedure TMIPSimplex.CopiarDefinicionesEnteras(MSPX: TMipSimplex);
var
  i, j: integer;
begin
  lstvents:= copy( MSPX.lstvents ) ;
  setlength( lstAcoplesVEnts, length( MSPX.lstAcoplesVEnts ) );
  for i := 0 to high( lstAcoplesVEnts ) do
  begin
      setLength(lstAcoplesVEnts[i], length( MSPX.lstAcoplesVEnts[i] ) );
      for j:= 0 to high( lstAcoplesVEnts[i] ) do
      begin
        lstAcoplesVEnts[i][j].ivar := MSPX.lstAcoplesVEnts[i][j].ivar;
        lstAcoplesVEnts[i][j].ires := MSPX.lstAcoplesVEnts[i][j].ires;
      end;
  end;
end;

procedure TMIPSimplex.set_entera(ivae, ivar: integer; CotaSup: integer);
begin
  lstvents[ivae - 1] := ivar;
  cota_sup_set(ivar, CotaSup);
end;

procedure TMIPSimplex.set_EnteraConAcople(ivae, ivar: integer;
  CotaSup: integer; ivarAcoplada, iresAcoplada: integer);
begin
  lstvents[ivae - 1] := ivar;
  cota_sup_set(ivar, CotaSup);
  setLength(lstAcoplesVEnts[ivae - 1], 0);
  SetLength(lstAcoplesVEnts[ivae - 1], 1);
  lstAcoplesVEnts[ivae - 1][0].ivar := ivarAcoplada;
  lstAcoplesVEnts[ivae - 1][0].ires := iresAcoplada;
end;

procedure TMIPSimplex.set_EnteraConAcoples(ivae, ivar: integer;
  CotaSup: integer; lstAcoples: TListaAcoplesVEntera);
begin
  lstvents[ivae - 1] := ivar;
  cota_sup_set(ivar, CotaSup);
  setLength(lstAcoplesVEnts[ivae - 1], 0);
  lstAcoplesVEnts[ivae - 1] := lstAcoples;
end;

procedure TMIPSimplex.fijarCotasSupDeVariablesAcopladas;
var
  i, j: integer;
  acoples: TListaAcoplesVEntera;
  xsup: NReal;
begin
  acoples := nil;
  for i := 0 to high(lstAcoplesVEnts) do
  begin
    acoples := lstAcoplesVEnts[i];
    if length(acoples) > 0 then
    begin
      for j := 0 to high(acoples) do
      begin
        if (acoples[j].ivar <> -1) and (self.flg_x[acoples[j].ivar] = 0) then
        begin
          xsup := pm[acoples[j].ires].pv[lstvents[i]] *
            (self.x_sup.pv[lstvents[i]] + self.x_inf.pv[lstvents[i]]) * 1.1;
          cota_sup_set(acoples[j].ivar, xsup);
        end;
      end;
    end;
  end;
end;


procedure TMIPSimplex.DumpSistemaToXLT(var f: textfile);
var
  k, j: integer;
begin
{$IFDEF DBG_CONTAR_CNT_SIMPLEX}
  if usimplex.cnt_debug >= usimplex.minCnt_DebugParaDump then
  begin
{$ENDIF}
    Write(f, 'NEnteras:', #9, length(lstvents));
    for k := 0 to high(lstvents) do
      Write(f, #9, lstvents[k]);
    writeln(f);

    writeln(f, 'ivae', #9, 'VarAcoplada', #9, 'ResAcoplada');
    for k := 0 to high(lstAcoplesVEnts) do
    begin
      if high(lstAcoplesVEnts[k]) >= 0 then
        for j := 0 to high(lstAcoplesVEnts[k]) do
          writeln(f, k, #9, lstAcoplesVEnts[k][j].ivar, #9, lstAcoplesVEnts[k][j].ires)
      else
        writeln(f, k, #9, -1, #9, -1);
    end;


    inherited DumpSistemaToXLT(f);
    if ElMejorNodoFactible <> nil then
    begin
      writeln(f, '--dump de ElMejorNodoFactible--');
      ElMejorNodoFactible.spx.DumpSistemaToXLT(f);

      for k := 1 to nc - 1 do
         write( f,  #9+ FloatToStr( xval(k) ) );
       writeln(f);
    end;
{$IFDEF DBG_CONTAR_CNT_SIMPLEX}
  end;
{$ENDIF}
end;

procedure TMIPSimplex.DumpSistemaToBIN(f: TStream);
var
  k, j, n: integer;
begin
  f.Write( nvents, SizeOf( nvents ) );
  for k:= 0 to high( lstvents ) do
     f.Write( lstvents[k], sizeOf( integer ) );

  for k := 0 to high(lstAcoplesVEnts) do
  begin
    n:= length( lstAcoplesVEnts[k] );
    f.Write( n, sizeOf( n ) );
    for j := 0 to high(lstAcoplesVEnts[k]) do
    begin
      f.Write( lstAcoplesVEnts[k][j].ivar, sizeOf( integer ) );
      f.write( lstAcoplesVEnts[k][j].ires, sizeOf( integer ) );
    end
  end;
  inherited DumpSistemaToBIN(f);
end;

constructor TMIPSimplex.CreateFromBin(f: TStream);
var
  k, j: integer;
  n: integer;
  kEntera: integer;

begin
  f.Read( nvents, sizeOf( nvents ) );
  setlength(lstvents, nvents);
  for k:= 0 to high( lstvents ) do
     f.Read( lstvents[k], sizeOf( integer ) );

  setLength(lstAcoplesVEnts, nvents);
  for k := 0 to high(lstAcoplesVEnts) do
  begin
    f.Read( n, sizeOf( n ) );
    setlength( lstAcoplesVEnts[k], n );
    for j := 0 to high(lstAcoplesVEnts[k]) do
    begin
      f.Read( lstAcoplesVEnts[k][j].ivar, sizeOf( integer ) );
      f.Read( lstAcoplesVEnts[k][j].ires, sizeOf( integer ) );
    end
  end;

  inherited CreateFromBin(f);

end;

function TMIPSimplex.EsEntera(j: integer): boolean;
var
  k: integer;
begin
  Result:= false;
  for k:= 0 to high( lstvents ) do
    if lstvents[k] = j then
    begin
      result:= true;
      break;
    end;
end;

{$IFDEF SIMSEE_GPUS}
procedure TMIPSimplex.toStruct(var struct : TSimplexGPUs);
var
  //res:TSimplexGPUs;
  kVariable, NVariables, kRestriccion, NRestricciones, kVariableEntera,i,j: Integer;
  cantcotasup, cantColumnas, cantFilas, restriccionesNoFijas : Integer;
begin

   writeln( 'ENTRO TOSTRUCT tabloide!');

  //4 (datos de control) + Largo 3 (vars) + 3 (cotas sup) + 4 (restricciones <=) + 1 (restricciones =) = 15
  //cantColumnas = 4 + NVariables + NRestricciones + cantcotasup

  cantcotasup:=0;


  NVariables:=nc - 1;
  NRestricciones:=nf - 1;
  writeln( 'NVariables: ',NVariables);
  writeln( 'NRestricciones: ',NRestricciones);

  for kVariable:=0 to NVariables-1 do
  begin
	if (self.x_sup.pv[kVariable+1]  > 0) then
		cantcotasup := cantcotasup + 1;
  end;

  //restriccionesNoFijas := NRestricciones - cnt_varfijas;
  //writeln(' restriccionesNoFijas :', restriccionesNoFijas);

  cantColumnas := 4 + NVariables + NRestricciones + cantcotasup;
  cantFilas := 6 + NRestricciones + cantcotasup ;

  writeln('cantcotasup :', cantcotasup);
  writeln('cantColumnas :', cantColumnas);
  writeln('cantFilas :', cantFilas);

  //cargo matriz con ceros
  for i:=0 to cantColumnas * cantFilas  do
	struct[i] := 0;



  //cargo los primeros 2 lugares de la matriz y funcion z a primera fila en la matriz
  struct[0]:= NVariables;
  struct[1]:= NRestricciones;
  for kVariable:=0 to NVariables -1 do
  begin
	struct[ kVariable + 4]:=e(NRestricciones + 1, kVariable + 1);
  end;


  //cargo segunda fila flg_x
  for kVariable:=0 to NVariables -1 do
  begin
	struct[(cantColumnas) + kVariable + 4]:= self.flg_x[kVariable+1];
  end;


  //cargo tercera fila x_sup
  for kVariable:=0 to NVariables-1 do
  begin
	struct[2*(cantColumnas) + kVariable +4]:= self.x_sup.pv[kVariable+1];
  end;


  //cargo cuarta fila x_inf
  for kVariable:=0 to NVariables-1 do
  begin
	struct[3*(cantColumnas) + kVariable +4]:= self.x_inf.pv[kVariable+1];
  end;


  //cargo Restricciones
  for kRestriccion:=0 to NRestricciones - 1  do
  begin
	struct[(kRestriccion+5)*(cantColumnas)+1 ]:= self.flg_y[kRestriccion+1]; //cargo columna flag_y
	struct[(kRestriccion+5)*(cantColumnas)+3 ]:= e(kRestriccion + 1 , NVariables + 1);//cargo Xb
    for kVariable:=0 to NVariables -1 do
	begin
		struct[(kRestriccion+5)*(cantColumnas) + kVariable + 4]:= e(kRestriccion + 1 , kVariable + 1);
	end;
  end;

  //print
  for j:=0 to cantFilas -1 do
  begin
	for i := 0 to cantColumnas -1 do
	begin
		write(' ', struct[(j*cantColumnas) + i]);
    end;
    writeln(' ');
    writeln(' ');
  end;
(*
var
  //res:TSimplexGPUs;
  kVariable, NVariables, kRestriccion, NRestricciones, kVariableEntera: Integer;
begin
  struct.NEnteras:= length(lstvents);
  struct.NVariables:= nc - 1;
  struct.NRestricciones:= nf - 1;
  struct.cnt_varfijas:=cnt_varfijas;
  struct.cnt_RestriccionesRedundantes:=cnt_RestriccionesRedundantes;

  NVariables:=struct.NVariables;

  struct.x_inf := @x_inf.pv[1];
  struct.x_sup := @x_sup.pv[1];

  struct.flg_x := @flg_x[1];
  struct.top := @top[1];

  NRestricciones:=struct.NRestricciones;

  struct.flg_y := @flg_y[1];
  struct.left := @left[1];

  struct.lstvents := @lstvents[0]; // lstvents y lstAcoplesVEnts indexan en 0

  for kVariableEntera:=0 to NVEnts-1 do
  begin
 	// El cero del acceso corresponde al unico acople que tiene esta sala
	// Falta resolver el caso donde puede variar la cantidad de acoples de una 
	// variable entera
	// Luego 0 para ivar
	//       1 para ires 
    struct.lstAcoplesVEnts[kVariableEntera*2 + 0] := lstAcoplesVEnts[kVariableEntera][0].ivar;
    struct.lstAcoplesVEnts[kVariableEntera*2 + 1] := lstAcoplesVEnts[kVariableEntera][0].ires;
  end;  

  for kRestriccion:=0 to NRestricciones do
  begin
    for kVariable:=0 to NVariables do
      struct.mat[kRestriccion*(NVariables+1) + kVariable]:=e(kRestriccion + 1, kVariable + 1);
  end;
*)
end;

procedure TMIPSimplex.loadFromStruct(struct: TSimplexGPUsOLD);
var
  kVariable, NVariables, kRestriccion, NRestricciones, kVariableEntera: Integer;
  iflg_x, iflg_y : ^ShortInt;
  itop, ileft : ^Integer;
begin

  //res.NEnteras:= length(lstvents);
  //res.NVariables:= nc - 1;
  //res.NRestricciones:= nf - 1;
  cnt_varfijas:=struct.cnt_varfijas;
  cnt_RestriccionesRedundantes:=struct.cnt_RestriccionesRedundantes;

  NVariables:=struct.NVariables;

  //SetLength(x_inf,NVariables);
  //SetLength(x_sup,NVariables);
  //SetLength(flg_x,NVariables);
  //SetLength(top,NVariables);
  iflg_x := struct.flg_x;
  itop := struct.top;
  for kVariable:=0 to NVariables - 1 do
  begin
    //x_inf.pon_e(kVariable + 1,struct.x_inf[kVariable]);
    //x_sup.pon_e(kVariable + 1,struct.x_sup[kVariable]);
    self.flg_x[kVariable + 1] := iflg_x^;
    Inc(iflg_x);
    self.top[kVariable + 1] := itop^;
    Inc(itop);
  end;

  NRestricciones:=struct.NRestricciones;

  //SetLength(flg_y,NRestricciones);
  //SetLength(left,NRestricciones);
  iflg_y := struct.flg_y;
  ileft := struct.left;
  for kRestriccion:=0 to NRestricciones - 1 do
  begin
    self.flg_y[kRestriccion + 1] := iflg_y^;
    Inc(iflg_y);
    self.left[kRestriccion + 1] := ileft^;
    Inc(ileft);
  end;


  for kVariableEntera:=0 to NVEnts-1 do
  begin
    lstAcoplesVEnts[kVariableEntera][0].ivar := struct.lstAcoplesVEnts[kVariableEntera*2 + 0];
    lstAcoplesVEnts[kVariableEntera][0].ires := struct.lstAcoplesVEnts[kVariableEntera*2 + 1];
  end;
  
//  SetLength(res.mat,NRestricciones + 1);
  for kRestriccion:=0 to NRestricciones do
  begin
  //  SetLength(res.mat[kRestriccion],NVariables + 1);
    for kVariable:=0 to NVariables do
      pon_e(kRestriccion + 1, kVariable + 1, struct.mat[kRestriccion*(NVariables+1) + kVariable]);
  end;
end;

procedure TMIPSimplex.loadFromOurStruct(struct: TSimplexGPUsOLD);
var
  kVariable, NVariables, kRestriccion, NRestricciones, kVariableEntera: Integer;
  iflg_x, iflg_y : ^ShortInt;
  itop, ileft : ^Integer;
begin

  //res.NEnteras:= length(lstvents);
  //res.NVariables:= nc - 1;
  //res.NRestricciones:= nf - 1;
  cnt_varfijas:=struct.cnt_varfijas;
  cnt_RestriccionesRedundantes:=struct.cnt_RestriccionesRedundantes;

  NVariables:=struct.NVariables;

  //SetLength(x_inf,NVariables);
  //SetLength(x_sup,NVariables);
  //SetLength(flg_x,NVariables);
  //SetLength(top,NVariables);
  iflg_x := struct.flg_x;
  itop := struct.top;
  for kVariable:=0 to NVariables - 1 do
  begin
    //x_inf.pon_e(kVariable + 1,struct.x_inf[kVariable]);
    //x_sup.pon_e(kVariable + 1,struct.x_sup[kVariable]);
    self.flg_x[kVariable + 1] := iflg_x^;
    Inc(iflg_x);
    self.top[kVariable + 1] := itop^;
    Inc(itop);
  end;

  NRestricciones:=struct.NRestricciones;

  //SetLength(flg_y,NRestricciones);
  //SetLength(left,NRestricciones);
  iflg_y := struct.flg_y;
  ileft := struct.left;
  for kRestriccion:=0 to NRestricciones - 1 do
  begin
    self.flg_y[kRestriccion + 1] := iflg_y^;
    Inc(iflg_y);
    self.left[kRestriccion + 1] := ileft^;
    Inc(ileft);
  end;


  for kVariableEntera:=0 to NVEnts-1 do
  begin
    lstAcoplesVEnts[kVariableEntera][0].ivar := struct.lstAcoplesVEnts[kVariableEntera*2 + 0];
    lstAcoplesVEnts[kVariableEntera][0].ires := struct.lstAcoplesVEnts[kVariableEntera*2 + 1];
  end;

//  SetLength(res.mat,NRestricciones + 1);
  for kRestriccion:=0 to NRestricciones do
  begin
  //  SetLength(res.mat[kRestriccion],NVariables + 1);
    for kVariable:=0 to NVariables do
      pon_e(kRestriccion + 1, kVariable + 1, struct.mat[kRestriccion*(NVariables+1) + kVariable]);
  end;
end;
{$ENDIF}

function TMIPSimplex.trySolve: boolean;
begin
  if ElNodoRaiz <> nil then
    ElNodoRaiz.Free;
  ElMejorNodoFactible := nil;
  fijarCotasSupDeVariablesAcopladas;
  ElNodoRaiz := TFichaNodoProblema.Create_Raiz(Self  {$IFDEF MIPSIMPLEXFROMORIGINAL}, Self{$ENDIF}, lstvents, lstAcoplesVEnts);
  result:=  ElNodoRaiz.Solve(ElNodoRaiz, ElMejorNodoFactible, '');
end;


function TMIPSimplex.SimplexSolucion: TSimplex;
begin
  if ElMejorNodoFactible <> nil then
    Result := ElMejorNodoFactible.spx
  else
    Result := nil;
end;

function TMIPSimplex.xval(ix: integer): NReal;
begin
  Result := ElMejorNodoFactible.spx.xval(ix);
end;

function TMIPSimplex.yval(iy: integer): NReal;
begin
  Result := ElMejorNodoFactible.spx.yval(iy);
end;

function TMIPSimplex.xmult_caja(ix: integer; var aflg_x: shortint): NReal;
begin
  Result:=ElMejorNodoFactible.spx.xmult_caja(ix, aflg_x);
end;

function TMIPSimplex.xmult(ix: integer): NReal;
begin
  Result := ElMejorNodoFactible.spx.xmult(ix);
end;

function TMIPSimplex.ymult(iy: integer): NReal;
begin
  Result := ElMejorNodoFactible.spx.ymult(iy);
end;

function TMIPSimplex.fval: NReal;
begin
  Result := ElMejorNodoFactible.spx.fval;
end;

function TMIPSimplex.resolver: integer;
var
  flgFallo: boolean;


begin
{$IFDEF DBG_CNT_MIPSPX}
inc( CNT_MIPSPX );
writeln( 'CNT_MIPSPX: ', CNT_MIPSPX );
(*
  if CNT_MIPSPX = 21433 then
begin
  DumpSistemaToXLT_('MIPSPX_21433.xlt', 'absurdo xentera_15' );
  DumpSistemaToBIN_archi('MIPSPX_21433.bin');
end;
*)
{$ENDIF}
  flgFallo:= not trySolve;
  // MAP: comento esto para solo tire el simplex
  (*
  {$IFDEF TRATAR_NORMALIZAR_SPX}
  if flgFallo then
  begin
    writeln( 'Fallo SPX_normal ... tratar_normalizar' );
    flgFallo := Tratar_Normalizar_SPX( self ) <> 0;
  end;
  {$ENDIF}

  {$IFDEF TRATAR_GLPK}
  if flgFallo then
  begin
    writeln( 'Fallo SPX_normalizado ... tratar_glpk' );
    flgFallo:= Tratar_GLPK( Self ) <> 0;
  end;
  {$ENDIF}
  *)
  // MAP: END
  if flgFallo then
    Result:= -1
  else
    Result:= 0;

{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
  writeln('Nodos Restantes: ', (nodos_Totales - (nodos_Recorridos +
    acum_Nodos_Recorridos)): 8: 0);
{$ENDIF}
end;


procedure TFichaNodoProblema.PrintSolEncabs(var f: textfile);
var
  k: integer;
begin
  Write(f, 'fval');
  for k := 1 to high(x) do
    Write(f, #9, 'x', k);
{  for k:= 1 to high( y ) do
    write( f, #9,'y',k);
  for k:= 1 to high( xmult ) do
    write( f, #9, 'mx',k );
  for k:= 1 to high( ymult ) do
    write( f, #9, 'my', k );}
  writeln(f);
end;

procedure TFichaNodoProblema.PrintSolVals(var f: Textfile);
var
  k: integer;
begin
  Write(f, fval);
  for k := 1 to high(x) do
    Write(f, #9, x[k]);
{  for k:= 1 to high( y ) do
    write( f, #9, y[k] );
  for k:= 1 to high( xmult ) do
    write( f, #9, xmult[k] );
  for k:= 1 to high( ymult ) do
    write( f, #9, ymult[k] );}
  writeln(f);
end;

function TFichaNodoProblema.primerCerradoEnLaCadenaHaciaArriba: TFichaNodoProblema;
var
  aux: TFichaNodoProblema;
begin
  aux := self;
  while (aux.padre <> nil) and (aux.Padre.arbolCerrado) do
    aux := aux.Padre;
  Result := aux;
end;

function TFichaNodoProblema.GetFichaRaiz: TFichaNodoProblema;
begin
  if Padre <> nil then
    result:= Padre.GetFichaRaiz
  else
    result:= self;
end;

{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
function TFichaNodoProblema.cnt_Relajaciones_Hacia_Abajo: ContadorParaCombinatoria;
var
  res: ContadorParaCombinatoria;
  i, nVarsNoFijadasRecorridas: integer;
begin
  res := 1;
  nVarsNoFijadasRecorridas := 0;
  i := 0;
  while nVarsNoFijadasRecorridas < nVarsNoFijadas do
  begin
    if abs(spx.flg_x[lstvents[i]]) <> 2 then
    begin
   //      res:= res * (spx.x_sup.e(lstvents[i]) + 1);
      res := res * (spx.x_sup.pv[lstvents[i]] + 1);
      Inc(nVarsNoFijadasRecorridas);
    end;
    Inc(i);
  end;
  Result := res;
end;

{$ENDIF}

constructor TFichaNodoProblema.Create_Raiz(xspx
   {$IFDEF MIPSIMPLEXFROMORIGINAL}
  , xspxOriginal {$ENDIF}: TSimplex;
  xlstvents: TDAOfNInt; xlstAcoplesVEnts: TDAOfAcoplesVEnts; flgClone: boolean);
begin
  inherited Create;
  arbolCerrado := False;
  Padre := nil;
  {$IFDEF MIPSIMPLEXFROMORIGINAL}
  spxOriginal:= xspxOriginal;
  {$ENDIF}
  if flgClone then
  begin
  {$IFDEF MIPSIMPLEXFROMORIGINAL}
    spx := TSimplex.Create_clone( spxOriginal );
    spx.x_inf.Copy( xspx.x_inf );
    spx.x_sup.Copy( xspx.x_sup );
    vcopy( spx.flg_x, xspx.flg_x );
    vcopy( spx.flg_y, xspx.flg_y );
  {$ELSE}
    spx := TSimplex.Create_clone(xspx)
  {$ENDIF}
  end
  else
    spx:= xspx;

  res_spx := -1010; // no corrido.
  Estado := CN_NoAsignado;
  lstvents := xlstvents;
  lstAcoplesVEnts := xlstAcoplesVEnts;
{$IFDEF RELAJACIONENCADENA}
  SetLength(relajadas, length(lstvents));
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
  nVarsNoFijadas := length(lstvents);
  nodos_Totales := cnt_Relajaciones_Hacia_Abajo;
  nodos_Recorridos := 0;
  acum_Nodos_Recorridos := 0;
{$ENDIF}
  kDesrelajar := -1;
  sentido := 0;
  ires := 0.5;
  nodo_izq := nil;
  nodo_der := nil;
end;

constructor TFichaNodoProblema.Create_Desrelajando(xPadre: TFichaNodoProblema;
  xspx  {$IFDEF MIPSIMPLEXFROMORIGINAL}, xspxOriginal{$ENDIF}: TSimplex; xlstvents: TDAOfNInt;
  xlstAcoplesVEnts: TDAOfAcoplesVEnts; xkEnteraDesrelajar: integer;
  xires: NReal; xsentido: integer);

  procedure resolverAcoples;
  var
    iAcoples{, miresY, mivarB, mivarA}: integer;
    Acoples: TListaAcoplesVEntera;
  begin
    Acoples := lstAcoplesVEnts[xkEnteraDesrelajar];
    for iAcoples := 0 to high(Acoples) do
    begin
      if Acoples[iAcoples].ivar <> -1 then
      begin
        spx.FijarVariable(Acoples[iAcoples].ivar, 0);
        spx.declararRestriccionRedundante(Acoples[iAcoples].ires);
      end;
    end;
  end;

var
  xmin, xmax: integer;
  xres: integer;
  cntVarFijas: integer;
  kVar, kRes: integer;

{$IFDEF SPXCONLOG}
  sdbg: string;
{$ENDIF}
begin
  inherited Create;
  {$IFDEF MIPSIMPLEXFROMORIGINAL}
  spxOriginal:= xspxOriginal;
  {$ENDIF}
  arbolCerrado := False;
  Padre := xPadre;
  nodo_izq := nil;
  nodo_der := nil;

{$IFDEF MIPSIMPLEXFROMORIGINAL}
  spx := TSimplex.Create_Clone( spxOriginal );
  cntVarFijas:= 0;
  for kVar:= 1 to spx.x_inf.n do
  begin
    if xspx.x_inf.e( kVar ) <> spx.x_inf.e( kVar ) then
        spx.cota_inf_set( kVar, xspx.x_inf.e( kVar ) );
    if xspx.x_sup.e( kVar ) <> spx.x_sup.e( kVar ) then
       spx.cota_sup_set( kVar, xspx.x_sup.e( kVar ) + xspx.x_inf.e( kVar ) );

    spx.flg_x[kVar] := abs( xspx.flg_x[kVar] );
    if ( spx.flg_x[kVar] = 2 ) then
     inc( cntVarFijas  );
  end;

  spx.cnt_varfijas:= cntVarFijas;
  for kRes:= 0 to high( spx.flg_y ) do
    spx.flg_y[kRes]:= abs( xspx.flg_y[kRes] );
{$ELSE}
  spx := TSimplex.Create_Clone(xspx);
{$ENDIF}

  // rch@20130307.bugfix
  // spx.cnt_RestriccionesRedundantes_:= 0;
  // en lugar de esto acomodamos las resolución de ingualdades en el Simplex.

  res_spx := -1010; // no corrido.

  Estado := CN_NoAsignado;
  lstvents := xlstvents;
  lstAcoplesVEnts := xlstAcoplesVEnts;
{$IFDEF RELAJACIONENCADENA}
  SetLength(relajadas, length(lstvents));
{$ENDIF}
  sentido := xsentido;
  ires:= xires;

  kDesrelajar := lstVEnts[xkEnteraDesrelajar];
  // bien ahora hacemos la desrelajación
  xmin := round(spx.x_inf.pv[kDesrelajar]);
  xmax := round(spx.x_sup.pv[kDesrelajar] + spx.x_inf.pv[kDesrelajar]);

  // rch@20140727 OJO, agrego esto pues venía ires en -0.18 y causa
  // loop de desrelajaciones con xmin = 0 y xmax = 1
  if xmax -xmin = 1 then
  begin
    if sentido < 0 then
      xmax := xmin
    else
      xmin := xmax;
  end
  else
  begin
    xres := round( xires );
    if sentido < 0 then
      xmax := max(xmin, min( xres, xmax-1 ))
    else
      xmin := min(xmax, max( xres, xmin+1 ));
  end;


{$IFDEF SPXCONLOG}
  spxActivo := spx;
  if spx.dbg_on then
  begin
    sdbg := 'MIPSpx Cambio de Rama';
    spx.writelog(sdbg);
  end;
{$ENDIF}

  if xmax = xmin then
  begin
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    self.nVarsNoFijadas := xPadre.nVarsNoFijadas - 1;
{$ENDIF}
    spx.FijarVariable(kDesrelajar, xmin);

{$IFDEF SPXCONLOG}
    if spx.dbg_on then
    begin
      sdbg := 'MIPSpx Fijar_Variable_Entera x: ' + spx.nombreVars[kDesrelajar] +
        ' valor= ' + IntToStr(xmin);
      spx.writelog(sdbg);
    end;
{$ENDIF}
    if xmin = 0 then
      resolverAcoples;
  end
  else
  if sentido < 0 then
  begin
    spx.cota_sup_set(kDesrelajar, xmax);
{$IFDEF SPXCONLOG}
    if spx.dbg_on then
    begin
      sdbg := 'MIPSpx Desrelajar_Variable_Entera x: ' +
        spx.nombreVars[kDesrelajar] + ' cotaSup= ' +
        IntToStr(round(spx.x_sup.pv[kDesrelajar])) + '->' + IntToStr(xmax);
      spx.writelog(sdbg);
    end;
{$ENDIF}
  end
  else
  begin
    spx.cota_inf_set(kDesrelajar, xmin);
{$IFDEF SPXCONLOG}
    if spx.dbg_on then
    begin
      sdbg := 'MIPSpx Desrelajar_Variable_Entera x: ' +
        spx.nombreVars[kDesrelajar] + ' cotaInf= ' +
        IntToStr(round(spx.x_inf.pv[kDesrelajar])) + '->' + IntToStr(xmin);
      spx.writelog(sdbg);
    end;
{$ENDIF}
  end;
end;

destructor TFichaNodoProblema.destroy;
begin
  setlength(x, 0);
{  setlength(y,0);
  setlength(xmult,0);
  setlength(ymult, 0 );}
{$IFDEF RELAJACIONENCADENA}
  SetLength(relajadas, 0);
{$ENDIF}
  if nodo_izq <> nil then
  begin
    nodo_izq.Free;
    nodo_izq := nil;
  end;

  if nodo_der <> nil then
  begin
    nodo_der.Free;
    nodo_der := nil;
  end;
  if spx <> nil then
  begin
    spx.Free;
    spx := nil;
  end;
  inherited destroy;
end;


function TFichaNodoProblema.Solve(ElNodoRaiz: TFichaNodoProblema;
  var ElMejorNodoFactible: TFichaNodoProblema; const Carpeta_ParaDump: string): boolean;
var
  primer_entera_relajada: integer;
  k: integer;
  RamaDelMejor: boolean;
  nodoAux, nodoAuxPadre: TFichaNodoProblema;
  aux: NReal;
{$IFDEF MIP_DBG}
  slog: string;
{$ENDIF}

begin
  try
    spxActivo := spx;
    res_spx := spxActivo.resolver;
    // MAP: Agrego/comento esto para solo tire el simplex
    //spxActivo := nil;
    if res_spx <> 0 then Result := false else Result := true;
    spxActivo.DumpSistemaToXLT_('sistemaResuelto_pascal_af.xlt','');
    Exit;
    // MAP: END
  except
    On E: Exception do
    begin
      spxActivo := nil;
      spx.DumpSistemaToXLT_(Carpeta_ParaDump + 'problema_.XLT', e.Message);
      raise E;
    end;
  end;

  if res_spx <> 0 then // rama infactible
  begin
{$IFDEF MIP_DBG}
    spx.writelog('MIP: Rama Infactible');
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    acum_Nodos_Recorridos := acum_Nodos_Recorridos + cnt_Relajaciones_Hacia_Abajo;
    if acum_Nodos_Recorridos > cantNodosSignificativa then
    begin
      nodos_Recorridos := nodos_Recorridos + acum_Nodos_Recorridos;
      acum_Nodos_Recorridos := 0;
    end;
    writeln('Nodos Restantes: ', (nodos_Totales -
      (nodos_Recorridos + acum_Nodos_Recorridos)): 8: 0);
{$ENDIF}
    Estado := CN_EliminadoPor_Infactible;
    Result := False; // eliminar rama
    exit;
  end;

  fval := spx.fval;
  if ElMejorNodoFactible <> nil then
  begin
    if fval <= ElMejorNodoFactible.fval then
      // esta rama se elimina por que no puede mejorar el costo
    begin
{$IFDEF MIP_DBG}
      spx.writelog('MIP: Rama Mayorada por la Factible');
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
      acum_Nodos_Recorridos := acum_Nodos_Recorridos + cnt_Relajaciones_Hacia_Abajo;
      if acum_Nodos_Recorridos > cantNodosSignificativa then
      begin
        nodos_Recorridos := nodos_Recorridos + acum_Nodos_Recorridos;
        acum_Nodos_Recorridos := 0;
      end;
      writeln('Nodos Restantes: ', (nodos_Totales -
        (nodos_Recorridos + acum_Nodos_Recorridos)): 8: 0);
{$ENDIF}
      Estado := CN_EliminadoPor_fval;
      Result := False; // eliminar rama
      exit;
    end;

{$IFDEF ABORTARAOPTIMIZACIONSINOESMEJORABLE}
      if  ( fval - ElMejorNodoFactible.fval ) <= ( ElNodoRaiz.fval - ElMejorNodoFactible.fval  )* fvalMargen  then
      begin
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
        acum_Nodos_Recorridos := acum_Nodos_Recorridos + cnt_Relajaciones_Hacia_Abajo;
        if acum_Nodos_Recorridos > cantNodosSignificativa then
        begin
          nodos_Recorridos := nodos_Recorridos + acum_Nodos_Recorridos;
          acum_Nodos_Recorridos := 0;
        end;
        writeln('Nodos Restantes: ',
          (
          nodos_Totales - (nodos_Recorridos + acum_Nodos_Recorridos)): 8: 0);
{$ENDIF}
        if Padre <> nil then
        begin
          arbolCerrado := True;
          spx.Free;
          spx := nil;
        end;
        Result := False;
        exit;
      end;
{$ENDIF}

  end;

  // creamos los vectores para guardar el resultado
  setlength(x, spx.nc);
  for k := 1 to spx.nc - 1 do
    x[k] := spx.xval(k);

  {$IFDEF MIP_DBG}
  slog:= 'MIP: x: ';
  for k := 1 to spx.nc - 1 do
    slog:= slog+ #9+ FloatToStr( x[k] );
  spx.writelog( slog );
  {$ENDIF}

{$IFDEF RELAJACIONENCADENA}
  // guardo cuales de mis variables estan relajadas
  for k := 0 to high(relajadas) do
    {$IFDEF SIN_BUGFIX_XC20201203}
    ents[k]])) > usimplex.CasiCero_VarEntera;
    {$ELSE}
    relajadas[k] := abs(round(x[lstvents[k]]) - x[lstvents[k]]) > usimplex.CasiCero_VarEntera;
    {$ENDIF}
  primer_entera_relajada := -1;
  // busco la primera variable entera relajada
  for k := 0 to high(relajadas) do
    if relajadas[k] then
    begin
      primer_entera_relajada := k;
      break;
    end;
  //ahora me fijo si alguna de las relajadas no estaba relajada en el padre
  if (primer_entera_relajada <> -1) and (padre <> nil) then
  begin
    for k := primer_entera_relajada to high(relajadas) do
      if relajadas[k] and not padre.relajadas[k] then
      begin
        primer_entera_relajada := k;
        break;
      end;
  end;
{$ELSE}
  // busco la primera variable entera relajada
  primer_entera_relajada := -1;
  for k := 0 to high(lstvents) do
  begin
    {$IFDEF SIN_BUGFIX_XC20201203}
    aux:=abs(frac( x[lstvents[k]] ));
    {$ELSE}
    aux:=abs(round(x[lstvents[k]]) - x[lstvents[k]]);  //@xc03122020 No reconocia como dominio factible a la solucion relajada.
    {$ENDIF}
    if aux > usimplex.CasiCero_VarEntera then
    begin
      primer_entera_relajada := k;
      break;
    end;
  end;
{$ENDIF}

  if primer_entera_relajada < 0 then
  begin
    Estado := CN_MejorFactible;
    if ElMejorNodoFactible <> nil then
    begin
      nodoAux := ElMejorNodoFactible.primerCerradoEnLaCadenaHaciaArriba;
      if nodoAux <> nil then
      begin
        nodoAuxPadre := nodoAux.Padre;
        if nodoAuxPadre <> nil then
        begin
          if nodoAuxPadre.nodo_izq = nodoAux then
            nodoAuxPadre.nodo_izq := nil
          else
            nodoAuxPadre.nodo_der := nil;
        end;
        nodoAux.Free;
      end;
    end;
    ElMejorNodoFactible := Self;
{$IFDEF MIP_DBG}
    spx.writelog('MIP: Encontramos un nuevo Mejor Nodo Factible');
{$ENDIF}
{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    acum_Nodos_Recorridos := acum_Nodos_Recorridos + cnt_Relajaciones_Hacia_Abajo;
    if acum_Nodos_Recorridos > cantNodosSignificativa then
    begin
      nodos_Recorridos := nodos_Recorridos + acum_Nodos_Recorridos;
      acum_Nodos_Recorridos := 0;
    end;
    writeln('Nodos Restantes: ', (nodos_Totales -
      (nodos_Recorridos + acum_Nodos_Recorridos)): 8: 0);
{$ENDIF}
    arbolCerrado := True;
    Result := True;
  end
  else
  begin
    Estado := CN_Relajado;



    kx_relajada := lstvents[primer_entera_relajada];
    {$IFDEF MIP_DBG}
        spx.writelog('MIP: Nodo Relajado Activo variable a desrelajar: kx_relajada:'
                           + IntToStr( kx_relajada )
                           +', primer_entera_relajada:'
                           + IntToStr( primer_entera_relajada ) );
    {$ENDIF}
    RamaDelMejor := False;


    //PA@25/05/07
    //Empezamos por el nodo derecho porque implica tener
    //prendidas todas las maquinas, lo que en general da un
    //costo menor que probar con todas las maquinas apagadas
    //y tener que usar las maquinas de falla. De esta forma
    //el primer nodo factible tiene un costo menor y permite
    //descartar mas nodos intermedios.
    nodo_der := TFichaNodoProblema.Create_Desrelajando(
      Self, spx  {$IFDEF MIPSIMPLEXFROMORIGINAL}, spxOriginal{$ENDIF}, lstvents, lstAcoplesVEnts, primer_entera_relajada, x[kx_relajada], 1);


    if not nodo_der.solve(ElNodoRaiz, ElMejorNodoFactible, Carpeta_ParaDump) then
    begin
      nodo_der.Free;
      nodo_der := nil;
    end
    else
      RamaDelMejor := True;


    nodo_izq := TFichaNodoProblema.Create_Desrelajando(
      Self, spx  {$IFDEF MIPSIMPLEXFROMORIGINAL}, spxOriginal{$ENDIF}, lstvents, lstAcoplesVEnts, primer_entera_relajada, x[kx_relajada], -1);
    if not nodo_izq.Solve(ElNodoRaiz, ElMejorNodoFactible, Carpeta_ParaDump) then
    begin
      nodo_izq.Free;
      nodo_izq := nil;
    end
    else
      RamaDelMejor := True;



{$IFDEF DBG_CONTAR_CNT_MIPSIMPLEX}
    acum_Nodos_Recorridos := acum_Nodos_Recorridos + 1;
    if acum_Nodos_Recorridos > cantNodosSignificativa then
    begin
      nodos_Recorridos := nodos_Recorridos + acum_Nodos_Recorridos;
      acum_Nodos_Recorridos := 0;
    end;
    //writeln('Nodos Restantes: ', (nodos_Totales - nodos_Recorridos - acum_Nodos_Recorridos):8:0);
{$ENDIF}
    // ya puedo liberar memoria si no soy el raiz
    if (Padre <> nil) then
    begin

      arbolCerrado := True;
      if (self <> ElMejorNodoFactible) then
      begin
        spx.Free;
        spx := nil;
      end;
    end;
    Result := RamaDelMejor;
  end;
end;

initialization
  spxActivo := nil;
  {$IFDEF DBG_CNT_MIPSPX}
  CNT_MIPSPX:= 0;
  {$ENDIF}

end.
