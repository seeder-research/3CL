package opencl

import (
       "bufio"
       "bytes"
       "io"
       "os"
       "regexp"

       "github.com/mumax/3cl/util"
)

func GenMergedKernelSource() string {
     // Build source code entirely
     var KernelsSource *bytes.Buffer

     grabKernelHeaders()

     // Write headers
     for _, codes := range Kernel_headers {
          KernelsSource.WriteString(codes.String())
     }

     // Write actual codes
     for _, codes := range Kernel_codes {
          KernelsSource.WriteString(codes.String())
     }

     return KernelsSource.String()
}

var ls_dir []string

func grabKernelHeaders() {
        // find .clh files
        if ls_dir == nil {
                dir, errd := os.Open(".")
                defer dir.Close()
                util.PanicErr(errd)
                var errls error
                ls_dir, errls = dir.Readdirnames(-1)
                util.PanicErr(errls)
        }

	idx := 0
        for _, f := range ls_dir {
                match, e := regexp.MatchString("..clh$", f)
                util.PanicErr(e)
                if match {
                        Kernel_headers[idx] = getHeaderFile(f)
			idx++
                }
        }

}

func getHeaderFile(fname string) *bytes.Buffer {
        f, err := os.Open(fname)
        util.PanicErr(err)
        defer f.Close()
        in := bufio.NewReader(f)
        var out bytes.Buffer
        out.Write(([]byte)("`"))
        line, err := in.ReadBytes('\n')
        for err != io.EOF {
                util.PanicErr(err)
                out.Write(line)
                line, err = in.ReadBytes('\n')
        }
        out.Write(([]byte)("`"))
        return &out
}
