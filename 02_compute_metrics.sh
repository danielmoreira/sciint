for i in $(cat ./image_list.txt); do
   echo 'start' $i;
   date;
   ./src/run.sh $i ./image_list.txt ''$i'.txt';
   date;
   echo 'end' $i;
done;
