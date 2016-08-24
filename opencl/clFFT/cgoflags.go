package clFFT

// This file provides CGO flags to find OpecnCL libraries and headers.

//#cgo darwin LDFLAGS: -framework clFFT
//#cgo !darwin LDFLAGS: -lclFFT
//
////default location:
//#cgo LDFLAGS:-L/usr/local/clFFT/lib64
//#cgo CFLAGS: -I/usr/local/clFFT/include
//
////Ubuntu 15.04::
//#cgo LDFLAGS:-L/usr/local/clFFT/lib64
//#cgo CFLAGS: -I/usr/local/clFFT/include
//
////arch linux:
//#cgo LDFLAGS:-L/opt/clFFT/lib64 -L/opt/clFFT/lib
//#cgo CFLAGS: -I/opt/clFFT/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/Intel/opencl/lib/x64
//#cgo windows CFLAGS: -IC:/Intel/opencl/include
import "C"

