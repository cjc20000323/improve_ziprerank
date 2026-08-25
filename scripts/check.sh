#!/bin/bash

for fd in /proc/8393/fd/*; do
    readlink "$fd"
done | sort -u | grep -E 'pytorch_model|\.bin|\.parquet|autodl'