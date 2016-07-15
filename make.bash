#! /bin/bash

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit || echo ""
ln -sf $(pwd)/post-commit .git/hooks/post-commit || echo ""

(cd opencl/kernels/cl && ./make.bash)  || exit 1
(cd opencl/kernels && ./make.bash)  || exit 1
go install -v github.com/mumax/3cl/cmd/... || exit 1
#go vet github.com/mumax/3/... || echo ""
#(cd test && mumax3 -vet *.mx3) || exit 1
#(cd doc && mumax3 -vet *.mx3)  || exit 1

