//go:build ignore
// +build ignore

// This program generates Go wrappers for opencl sources.
// The opencl file should contain exactly one __kernel void.

package main

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"regexp"
	"text/scanner"

	"github.com/mumax/3cl/opencl/kernels_src"
	"github.com/mumax/3cl/util"
)

// template data
type Kernel_stuff struct {
	OCL  map[string]string
	Code map[string]string
}

var ls_dirclh []string
var ls_dircl []string

func main() {
	// find .clh files
	if ls_dirclh == nil {
		dirclh, errd := os.Open("./clh")
		defer dirclh.Close()
		util.PanicErr(errd)
		var errls error
		ls_dirclh, errls = dirclh.Readdirnames(-1)
		util.PanicErr(errls)
	}

	// find .cl files
	if ls_dircl == nil {
		dircl, errd := os.Open("./cl")
		defer dircl.Close()
		util.PanicErr(errd)
		var errls error
		ls_dircl, errls = dircl.Readdirnames(-1)
		util.PanicErr(errls)
	}

	// get header codes in .clh files
	opencl_codes := &Kernel_stuff{make(map[string]string), make(map[string]string)}
	for _, f := range ls_dirclh {
		match, e := regexp.MatchString("..clh$", f)
		util.PanicErr(e)
		if match {
			fkey := f[:len(f)-len(".clh")]
			opencl_codes.OCL[fkey] = getFile("./clh/" + f)
		}
	}

	// get names of kernels available in .cl files
	for _, f := range ls_dircl {
		match, e := regexp.MatchString("..cl$", f)
		util.PanicErr(e)
		if match {
			kname := getKernelName("./cl/" + f)
			opencl_codes.Code[kname] = getFile("./cl/" + f)
		}
	}

	tmpBuffer := new(bytes.Buffer)
	tmpBuffer.WriteString("package kernels\n")
	tmpBuffer.WriteString("\n\n// THIS FILE WAS CREATED BY OPENCL2GO\n")
	tmpBuffer.WriteString("// MODIFYING THIS FILE IS FUTILE!!!!!\n\n")
	tmpBuffer.WriteString("func OpenclProgramSource() string {\n")
	tmpBuffer.WriteString("	opencl_codes := `\n")
	for _, keynames := range kernels_src.OCLHeadersList {
		tmpBuffer.WriteString(opencl_codes.OCL[keynames])
	}
	for _, keynames := range kernels_src.OCLKernelsList {
		tmpBuffer.WriteString(opencl_codes.Code[keynames])
	}
	tmpBuffer.WriteString("\n`\n\n	return opencl_codes\n}\n")

	wrapfname := "../kernels/program_wrapper.go"
	wrapout, err := os.OpenFile(wrapfname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.PanicErr(err)
	wrapout.WriteString(tmpBuffer.String())
	wrapout.Close()

	tmpBuffer = new(bytes.Buffer)
	tmpBuffer.WriteString("package opencl\n\n")
	tmpBuffer.WriteString("var KernelNames = []string{\n")
	for idx, keynames := range kernels_src.OCLKernelsList {
		if idx == len(kernels_src.OCLKernelsList)-1 {
			tmpBuffer.WriteString("\t\"" + keynames + "\"}\n")
		} else {
			tmpBuffer.WriteString("\t\"" + keynames + "\",\n")
		}
	}
	wrapfname = "../opencl_kernels_wrapper.go"
	wrapout, err = os.OpenFile(wrapfname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.PanicErr(err)
	wrapout.WriteString(tmpBuffer.String())
	wrapout.Close()
}

func getKernelName(fname string) string {
	// open opencl file
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()

	// read tokens
	var token []string
	var s scanner.Scanner
	s.Init(f)
	tok := s.Scan()
	for tok != scanner.EOF {
		if !filter(s.TokenText()) {
			token = append(token, s.TokenText())
		}
		tok = s.Scan()
	}

	// find function name and arguments
	funcname := ""
	for i := 0; i < len(token); i++ {
		if token[i] == "__kernel" {
			funcname = token[i+2]
			break
		}
	}
	return funcname
}

func getFile(fname string) string {
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()
	in := bufio.NewReader(f)
	var out bytes.Buffer
	line, err := in.ReadBytes('\n')
	for err != io.EOF {
		util.PanicErr(err)
		out.Write(line)
		line, err = in.ReadBytes('\n')
	}
	return out.String()
}

// should token be filtered out of stream?
func filter(token string) bool {
	switch token {
	case "__restrict":
		return true
	case "__global":
		return true
	case "__constant":
		return true
	case "__local":
		return true
	case "volatile":
		return true
	case "unsigned":
		return true
	case "signed":
		return true
	case "const":
		return true
	}
	return false
}
