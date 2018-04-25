#! /bin/bash

if [ -n "${GOPATH+set}" ]; then
        if [ -n "${GOPATH}" ]; then
        rm -frv ${GOPATH}/pkg/*
        binlist=`ls ./cmd`
        for fname in ${binlist}; do
                if [ -d "./cmd/${fname}" ]; then
                        if [ -f "${GOPATH}/bin/${fname}" ]; then
                                rm -fv ${GOPATH}/bin/${fname}
                        fi
                        if [ -f "${GOPATH}/bin/${fname}.exe" ]; then
                                rm -fv ${GOPATH}/bin/${fname}.exe
                        fi
                fi
        done
    fi
fi
rm -fv  opencl/*_wrapper.go
rm -fv  opencl/kernels_src/opencl2go
rm -fv  opencl/kernels_src/cl/ocl2go
rm -fv  opencl/kernels_src/*_wrapper.go
rm -fv  opencl/kernels/*_wrapper.go
