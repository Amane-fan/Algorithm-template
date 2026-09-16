#!/bin/bash

g++ -std=c++20 -include "./include/pch.hpp" $1.cpp -o $1

./$1 < $1.in > $1.out

cat $1.out