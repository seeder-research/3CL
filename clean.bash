#! /bin/bash

if [ -n "${GOPATH+set}" ]; then
        if [ -n "${GOPATH}" ]; then
        rm -frv $GOPATH/pkg/*
        rm -frv $GOPATH/bin/mumax3cl*
    fi
fi
rm -fv  opencl/*_wrapper.go
rm -fv  opencl/kernels_src/opencl2go
rm -fv  opencl/kernels_src/cl/ocl2go
rm -fv  opencl/kernels_src/*_wrapper.go
rm -fv  opencl/kernels/*_wrapper.go
