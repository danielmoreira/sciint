#!/usr/bin/env bash

bgn=$(date);
cd /provenance/io;

for i in $(ls -d -1 *.json); do
  echo 'python3.6 -u ../build_graph.py '$i'';
  python3.6 -u ../build_graph.py $i;
done;

echo $bgn;
date;
echo Acabou.;