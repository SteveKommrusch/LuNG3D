for i in ../L*; do mkdir $i/$1/pass7; mv $i/feede* $i/$1/pass7; done
for i in ../L*; do cp $i/python.out $i/rnd.png $i/step.png $i/final.png $i/$1/; done
touch Shapes.py
cp Shapes.py tail.py final.py $1
for i in ../L?; do cp Shapes.py tail.py final.py $i; done
for i in ../L*; do : > $i/feederr.csv; rm $i/img2.csv; done
for i in ../L*; do rm $i/$1/pass*/rnd.csv; gzip $i/$1/pass*/alnsb.out; done
python scrfeat.py ../*/$1/pass*/feat.csv
wc ../*/$1/pass*/feat.csv > $1/summary
for i in ../*/$1/pass*/python.out; do grep "INFO: iteration" /dev/null $i | tail -n 1; done >> $1/summary 
for i in ../*/$1/pass*/python.out; do grep "generated: 420, clean" $i /dev/null | tail -n 1; done >> $1/summary
for i in ../L*/$1/pass*/err.csv; do perl -e 'while (<>) {$s=$s+$_*1000}; printf("$ARGV: %.5f\n",$s/$.)' $i; done >> $1/summary 
for i in ../L*/$1/pass*/feederr.csv; do perl -e 'while (<>) {$s=$s+$_*1000}; printf("$ARGV: %.5f\n",$s/$.)' $i; done >> $1/summary 
cat ../L*/$1/featdist.txt >> $1/summary
