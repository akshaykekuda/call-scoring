#!/bin/bash

# This script removes space in file names
for f in "$1"*.opus; do mv "$f" `echo $f | tr ' ' '_'`; done
