package opencl

import (
	"testing"

)

func TestFFTCalcChirpLength(t *testing.T) {
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

