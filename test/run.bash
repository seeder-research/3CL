#! /bin/bash

set -e

mumax3cl -vet *.mx3

mumax3cl -paranoid=false -failfast -cache /var/tmp/elefongx-1 -f -http "" *.go *.mx3

