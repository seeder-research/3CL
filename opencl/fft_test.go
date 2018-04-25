package opencl

import (
	"testing"

)

func TestFFTCalcChirpLength0(t *testing.T) {
	full_length := 68
	result := calcChirpLength(full_length)
	if result != 17 {
		t.Error("got:", result)
	}
	full_length = -4
	result = calcChirpLength(full_length)
	if result != -1 {
		t.Error("got:", result)
	}
	full_length = 2*3*4*5*7*8*11*13*23
	result = calcChirpLength(full_length)
	if result != 23 {
		t.Error("got:", result)
	}
}

func TestFFTCalcChirpLength1(t *testing.T) {
	length_x := 2*5*11
	length_y := 4*8
	length_z := 3*13
	flag, result := chkFFTSize(length_x, length_y, length_z)
	if flag != true {
		t.Error("got:", result)
	}
	length_x = 2*5*11*23
	length_y = 4*8*43
	length_z = 3*13*71
	flag, result = chkFFTSize(length_x, length_y, length_z)
	if flag != false {
		t.Error("got:", result)
	}
}

