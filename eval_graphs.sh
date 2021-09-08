#!/usr/bin/env bash

bgn=$(date);
cd /provenance/io;

echo 'python3.6 -u ../eval_graph.py ./';
python3.6 -u ../eval_graph.py ./;

echo $bgn;
date;
echo Acabou.;