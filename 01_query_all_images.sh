#!/usr/bin/env bash

bgn=$(date);
cd /ranking/io;

for i in $(cat ../ranking_data/image_list.txt); do
   echo 'python3.7 -u ../src/flat_query.py '$i' ../ranking_data/image_list.txt '$i'.txt';
   python3.7 -u ../src/flat_query.py $i ../ranking_data/image_list.txt ''$i'.txt';
done;

echo $bgn;
date;
echo Acabou.;
