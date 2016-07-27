#! /bin/bash

rm -frv $GOPATH/pkg/*
rm -frv $GOPATH/bin/*
rm -fv  opencl/*_wrapper.go
rm -fv  opencl/kernels_src/opencl2go
rm -fv  opencl/kernels_src/cl/ocl2go
rm -fv  opencl/kernels_src/*_wrapper.go
rm -fv  opencl/kernels/*_wrapper.go
