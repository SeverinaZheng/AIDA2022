#!/bin/bash


NAME="CPU"
python ../read_*Swi* $NAME 1000 80000&

sleep 4
NAME="GPU"
python ../read_*Swi* $NAME 180000 25000&

sleep 8
python ../../short_query.py

sleep 4
python ../../query.py
