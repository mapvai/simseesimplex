#!/bin/bash

rm ../../../../../ppus/testsimplexgpus/*

cd cuda/
rm -r *.o
nvcc -c ini_mem.cu resolver_cuda.cu free_mem.cu resolver_cuda_ccpu.cu

cd ../
/usr/bin/fpc @fp64.cfg testSimplexGPUs.lpr -otestSimplexGPUs.o -dVERBOSE