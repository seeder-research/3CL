#! /bin/bash

go build opencl2go.go || exit 1

./opencl2go || exit 1

