#! /bin/bash

go build ocl2go.go || exit 1

for f in *.ocl; do
	g=$(echo $f | sed 's/\.ocl$//') # file basename
	if [[ $f -nt $g'_wrapper.go' ]]; then
		./cuda2go $f || exit 1
	fi
done

