program testSimplexGPUs;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes,
  utestSimplexGPUs,
  uauxiliares, umipsimplex,
  usimplexgpus,
  sysutils
  { you can add units after this };

var
  NTrayectorias , kTrayectoria: Integer;
  simplex_array : TDAOfMIPSimplex;
  struct_simplex_array : TDAOfSimplexGPUs;
  d_simplex_array : ^TSimplexGPUs;
  h_simplex_array: TDAOfSimplexGPUs;
  X: QWord;

  pasres: Integer;
  simplex_pascal: TMIPSimplex;
  ccpures: Integer;
  simplex_ccpu: TMIPSimplex;
begin

  writeLn('INI INI');

  setSeparadoresGlobales;


  // Resolver pascal
  simplex_pascal := TMIPSimplex.CreateFromArchiXLT(ParamStr(1));
  pasres := simplex_pascal.resolver;
  writeLn('Pascal result: ' + IntToStr(pasres));
  //simplex_pascal.DumpSistemaToXLT_('sistemaResuelto_pascal.xlt',''); El DUMP se hace en usimplex jsuto despues de tirar el usimplex.resolver pq despues hace otras cosas y cambia los datos en la estructura
  writeLn('FREE pascal');
  simplex_pascal.Free;

  // Resolver cpu, 1 caso
  SetLength(struct_simplex_array, 1);

  simplex_ccpu := TMIPSimplex.CreateFromArchiXLT(ParamStr(1));

  SetLength(struct_simplex_array[0].lstAcoplesVEnts, 2*simplex_ccpu.NVEnts);
  SetLength(struct_simplex_array[0].mat, (simplex_ccpu.nf)*(simplex_ccpu.nc));

  simplex_ccpu.toStruct(struct_simplex_array[0]);

  d_simplex_array := nil;
  SetLength(h_simplex_array, 0);

  writeLn('INI RESOLVER CPU');
  resolver_cuda(struct_simplex_array, d_simplex_array, h_simplex_array, 1);
  writeLn('END RESOLVER CPU');

  writeLn('loadFromStruct');
  simplex_ccpu.loadFromStruct(struct_simplex_array[0]);
  writeLn('DumpSistemaToXLT');
  simplex_ccpu.DumpSistemaToXLT_('sistemaResuelto_ccpu.xlt','');
  writeLn('FREE');
  simplex_ccpu.Free;
  writeLn('FIN');

  (*
  // Resolver cpu, N casos

  NTrayectorias := ParamStr(ParamCount).ToInteger;
  SetLength(simplex_array, NTrayectorias);
  SetLength(struct_simplex_array, NTrayectorias);

  for kTrayectoria := 0 to NTrayectorias - 1 do
  begin
    simplex_array[kTrayectoria] := TMIPSimplex.CreateFromArchiXLT(ParamStr(1));

    SetLength(struct_simplex_array[kTrayectoria].lstAcoplesVEnts, 2*simplex_array[kTrayectoria].NVEnts);
    SetLength(struct_simplex_array[kTrayectoria].mat, (simplex_array[kTrayectoria].nf)*(simplex_array[kTrayectoria].nc));

    simplex_array[kTrayectoria].toStruct(struct_simplex_array[kTrayectoria]);
  end;

  d_simplex_array := nil;
  SetLength(h_simplex_array, 0);

  writeLn('INI RESOLVER CPU');
  resolver_cuda(struct_simplex_array, d_simplex_array, h_simplex_array, NTrayectorias);
  writeLn('END RESOLVER CPU');

  for kTrayectoria := 0 to NTrayectorias - 1 do
  begin
    writeLn('loadFromStruct'+IntToStr(kTrayectoria));
    simplex_array[kTrayectoria].loadFromStruct(struct_simplex_array[kTrayectoria]);
    writeLn('DumpSistemaToXLT_'+IntToStr(kTrayectoria));
    simplex_array[kTrayectoria].DumpSistemaToXLT_('sistemaResuelto_ccpu_'+IntToStr(kTrayectoria)+'.xlt','');
    writeLn('FREE'+IntToStr(kTrayectoria));
    simplex_array[kTrayectoria].Free;
  end;
  writeLn('FIN');
  *)

  (*
  // Resolver GPU
  for kTrayectoria := 0 to NTrayectorias - 1 do
  begin
    simplex_array[kTrayectoria] := TMIPSimplex.CreateFromArchiXLT(ParamStr(1));
	// Se alocan las memorias necesarias para las copias de la matriz y la lista de acoples
	// el resto son punteros
	SetLength(struct_simplex_array[kTrayectoria].lstAcoplesVEnts,2*simplex_array[kTrayectoria].NVEnts);
    SetLength(struct_simplex_array[kTrayectoria].mat,
            (simplex_array[kTrayectoria].nf)*(simplex_array[kTrayectoria].nc));
    simplex_array[kTrayectoria].toStruct(struct_simplex_array[kTrayectoria]);

    if (kTrayectoria = 0) then
    begin
      X := GetTickCount64;
      ini_mem(d_simplex_array,
              h_simplex_array,
              NTrayectorias,
              struct_simplex_array[kTrayectoria].NEnteras,
              struct_simplex_array[kTrayectoria].NVariables,
              struct_simplex_array[kTrayectoria].NRestricciones,
              struct_simplex_array[kTrayectoria].cnt_varfijas,
              struct_simplex_array[kTrayectoria].cnt_RestriccionesRedundantes);
      writeln('Ini GPU ',(GetTickCount64 - X));
    end;
  end;
  setSeparadoresLocales;

  resolver_cuda(struct_simplex_array, d_simplex_array, h_simplex_array, NTrayectorias);

  for kTrayectoria := 0 to NTrayectorias - 1 do
  begin
    simplex_array[kTrayectoria].loadFromStruct(struct_simplex_array[kTrayectoria]);
  end;

  for kTrayectoria := 0 to NTrayectorias - 1 do
  begin
    {$IFDEF VERBOSE}
	writeLn('Resolver');
    {$ENDIF}
    simplex_array[kTrayectoria].resolver;
    {$IFDEF VERBOSE}
	writeLn('Dump');
    {$ENDIF}
    simplex_array[kTrayectoria].DumpSistemaToXLT_('sistemaResuelto_'+IntToStr(kTrayectoria)+'.xlt','');
    {$IFDEF VERBOSE}
	writeLn('FREE');
    {$ENDIF}
    simplex_array[kTrayectoria].Free;
  end;

  free_mem(d_simplex_array,
    h_simplex_array,
    NTrayectorias,
    struct_simplex_array[kTrayectoria].NEnteras,
    struct_simplex_array[kTrayectoria].NVariables,
    struct_simplex_array[kTrayectoria].NRestricciones,
    struct_simplex_array[kTrayectoria].cnt_varfijas,
    struct_simplex_array[kTrayectoria].cnt_RestriccionesRedundantes);
  *)
end.
