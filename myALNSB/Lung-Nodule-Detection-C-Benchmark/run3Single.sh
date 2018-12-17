echo "touch ~/p/LuNG/Shapes.py before starting python to insure sequencing"
while [ ~/p/LuNG/Shapes.py -nt ~/p/LuNG/step.png ]; do sleep 9; done
while [ -f ~/p/LuNG/img2.csv ] && [ ~/p/LuNG/img2.csv -nt ~/p/LuNG/step.png ]; do sleep 9; done
while [ ~/p/LuNG/Shapes.py -nt ~/p/La/step.png ]; do sleep 9; done
while [ -f ~/p/La/img2.csv ] && [ ~/p/La/img2.csv -nt ~/p/La/step.png ]; do sleep 9; done
while [ ~/p/LuNG/Shapes.py -nt ~/p/Lb/step.png ]; do sleep 9; done
while [ -f ~/p/Lb/img2.csv ] && [ ~/p/Lb/img2.csv -nt ~/p/Lb/step.png ]; do sleep 9; done
sleep 9

mkdir -p ~/p/LuNG/$1
cp ~/p/LuNG/bottleneck.csv ~/p/LuNG/feederr.csv ~/p/LuNG/err.csv ~/p/LuNG/rnd.csv ~/p/LuNG/python.out ~/p/LuNG/$1
grep -v "^0.000,0.599,0.000,0.599" ~/p/LuNG/rnd.csv > segInputOverride.csv
rm ~/p/LuNG/img2.csv
alnsb --show-environment --verbose-level 42 --thresolding_skip4 --rotate-90-right --enable-timing --rotation-skip --levelscale-skip > ~/p/LuNG/$1/alnsb.out
perl -ne 'if (s/\*\* Disp.*716 \*\*\n//) {$l=<>; $_.=$l; }; /KEEPCSV:/ && ($k=$k+1); ($k>20) && s/KEEPCSV: // && s/,$// && print' ~/p/LuNG/$1/alnsb.out > ~/p/LuNG/img2.csv
perl -ne 'if ($in==1) { s/\s+/,/g; /^([^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,)/ && ($l = "$1") && ($in=2) } else { ($in==2) && /distance to pos: (\d+\.\d+,).*neg: (\d+\.\d+)/ && (print "$l$1$2\n") && ($in=0); /^KEEPFEAT:/ && ($in = 1) }' ~/p/LuNG/$1/alnsb.out > ~/p/LuNG/$1/feat.csv

mkdir -p ~/p/La/$1
cp ~/p/La/bottleneck.csv ~/p/La/feederr.csv ~/p/La/err.csv ~/p/La/rnd.csv ~/p/La/python.out ~/p/La/$1
grep -v "^0.000,0.599,0.000,0.599" ~/p/La/rnd.csv > segInputOverride.csv
rm ~/p/La/img2.csv
alnsb --show-environment --verbose-level 42 --thresolding_skip4 --rotate-90-right --enable-timing --rotation-skip --levelscale-skip > ~/p/La/$1/alnsb.out
perl -ne 'if (s/\*\* Disp.*716 \*\*\n//) {$l=<>; $_.=$l; }; /KEEPCSV:/ && ($k=$k+1); ($k>20) && s/KEEPCSV: // && s/,$// && print' ~/p/La/$1/alnsb.out > ~/p/La/img2.csv
perl -ne 'if ($in==1) { s/\s+/,/g; /^([^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,)/ && ($l = "$1") && ($in=2) } else { ($in==2) && /distance to pos: (\d+\.\d+,).*neg: (\d+\.\d+)/ && (print "$l$1$2\n") && ($in=0); /^KEEPFEAT:/ && ($in = 1) }' ~/p/La/$1/alnsb.out > ~/p/La/$1/feat.csv

mkdir -p ~/p/Lb/$1
cp ~/p/Lb/bottleneck.csv ~/p/Lb/feederr.csv ~/p/Lb/err.csv ~/p/Lb/rnd.csv ~/p/Lb/python.out ~/p/Lb/$1
grep -v "^0.000,0.599,0.000,0.599" ~/p/Lb/rnd.csv > segInputOverride.csv
rm ~/p/Lb/img2.csv
alnsb --show-environment --verbose-level 42 --thresolding_skip4 --rotate-90-right --enable-timing --rotation-skip --levelscale-skip > ~/p/Lb/$1/alnsb.out
perl -ne 'if (s/\*\* Disp.*716 \*\*\n//) {$l=<>; $_.=$l; }; /KEEPCSV:/ && ($k=$k+1); ($k>20) && s/KEEPCSV: // && s/,$// && print' ~/p/Lb/$1/alnsb.out > ~/p/Lb/img2.csv
perl -ne 'if (s/\*\* Disp.*716 \*\*\n//) {$l=<>; $_.=$l; }; /KEEPCSV:/ && ($k=$k+1); ($k>20) && s/KEEPCSV: // && s/,$// && print' ~/p/La/$1/alnsb.out > ~/p/La/$1/img2.csv.dummy
perl -ne 'if ($in==1) { s/\s+/,/g; /^([^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,)/ && ($l = "$1") && ($in=2) } else { ($in==2) && /distance to pos: (\d+\.\d+,).*neg: (\d+\.\d+)/ && (print "$l$1$2\n") && ($in=0); /^KEEPFEAT:/ && ($in = 1) }' ~/p/Lb/$1/alnsb.out > ~/p/Lb/$1/feat.csv
sleep 9
cp ~/p/LuNG/$1/feat.csv ~/p/LuNG
cp ~/p/La/$1/feat.csv ~/p/La
rm ~/p/La/$1/img2.csv.dummy
sleep 9
cp ~/p/Lb/$1/feat.csv ~/p/Lb
