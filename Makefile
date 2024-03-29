
# Use the default go compiler
GO_BUILDFLAGS=-compiler gc
# Or uncomment the line below to use the gccgo compiler, which may 
# or may not be faster than gc and which may or may not compile...
# GO_BUILDFLAGS=-compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'

CGO_CFLAGS_ALLOW='(-fno-schedule-insns|-malign-double|-ffast-math)'


.PHONY: all clkernels clean realclean hooks


all: clkernels
	go install -v $(GO_BUILDFLAGS) github.com/mumax/3cl/cmd/...

clkernels:
	cd ./opencl && $(MAKE)

hooks: .git/hooks/post-commit .git/hooks/pre-commit

.git/hooks/post-commit: post-commit
	ln -sf $(CURDIR)/$< $@

.git/hooks/pre-commit: pre-commit
	ln -sf $(CURDIR)/$< $@

clean:
	rm -frv $(GOPATH)/pkg/*/github.com/mumax/3cl/*
	rm -frv $(GOPATH)/bin/mumax3*
	cd ./opencl && $(MAKE) clean

realclean: clean
	cd ./opencl && ${MAKE} realclean
