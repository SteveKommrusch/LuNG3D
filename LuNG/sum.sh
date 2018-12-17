python scrfeat.py ../*/$1/pass*/feat.csv
wc ../*/$1/pass*/feat.csv > $1/summary
for i in ../*/$1/pass*/python.out; do grep "INFO: iteration" /dev/null $i | tail -n 1; done >> $1/summary 
for i in ../*/$1/pass*/python.out; do grep "generated: 420, clean" $i /dev/null | tail -n 1; done >> $1/summary
for i in ../L*/$1/pass*/err.csv; do perl -e 'while (<>) {$s=$s+$_*1000}; printf("$ARGV: %.5f\n",$s/$.)' $i; done >> $1/summary 
for i in ../L*/$1/pass*/feederr.csv; do perl -e 'while (<>) {$s=$s+$_*1000}; printf("$ARGV: %.5f\n",$s/$.)' $i; done >> $1/summary 
cat ../L*/$1/featdist.txt >> $1/summary
