mumax3cl 
======

GPU accelerated micromagnetic simulator.


Downloads and documentation
---------------------------

http://mumax.github.io


Paper
-----

The Design and Verification of mumax3:

http://scitation.aip.org/content/aip/journal/adva/4/10/10.1063/1.4899186


Tools
-----

https://godoc.org/github.com/mumax/3/cmd


Building from source (for linux)
--------------------

Consider downloading a pre-compiled binary. If you want to compile nevertheless:

  * install the OpenCL driver, if not yet present.
   - if unsure, it's probably already there
   - requires OpenCL 1.2 support
  * install Go 
    - https://golang.org/dl/
    - set $GOPATH
  * install a C compiler
    - Ubuntu: `sudo apt-get install gcc`
    - MacOSX: https://developer.apple.com/xcode/download/
    - Windows: http://sourceforge.net/projects/mingw-w64/
  * if you have git installed: 
    - `go get github.com/mumax/3/cmd/mumax3`
  * if you don't have git:
    - seriously, no git?
    - get the source from https://github.com/mumax/3cl/releases
    - unzip the source into $GOPATH/src/github.com/mumax/3cl
    - `cd $GOPATH/src/github.com/mumax/3cl/cmd/mumax3`
    - `go install`
  * optional: install gnuplot if you want pretty graphs
    - Ubuntu: `sudo apt-get install gnuplot`

Your binary is now at `$GOPATH/bin/mumax3cl`

To do all at once on Ubuntu:
```
sudo apt-get install git golang-go gcc gnuplot
export GOPATH=$HOME go get -u -v github.com/mumax/3cl/cmd/mumax3cl
```

Contributing
------------

Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
