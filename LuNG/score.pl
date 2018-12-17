#!/usr/bin/perl
$p=$ARGV[0]; 
$p=~/pass(\d)/ || die "Need to include 'pass' number";
printf "Results                No    Twice          LuNG                       La                        Lb                        Lc\n";
printf "Directory              Fdbk  Fdbk   Acpt  MSE Dist MMSE  Tot  Acpt  MSE Dist MMSE  Tot  Acpt  MSE Dist MMSE  Tot  Acpt  MSE Dist MMSE  Tot\n";
printf "---------              ----- -----  ------------------------  ------------------------  ------------------------  -------------------------\n";
foreach $file (@ARGV) {
  if ( -e $file."/summary" ) { # only run check on directories that have summary
    open FILE , "<".$file."/summary" || die $!;
    while(<FILE>) { 
      /^\s+(\d+).*LuNG.[Ee][Xx][Pp].*$p.feat.csv/ && ($Au=($1-20)/400);
      /^\s+(\d+).*La.[Ee][Xx][Pp].*$p.feat.csv/ && ($Aa=($1-20)/400);
      /^\s+(\d+).*Lb.[Ee][Xx][Pp].*$p.feat.csv/ && ($Ab=($1-20)/400);
      /^\s+(\d+).*Lc.[Ee][Xx][Pp].*$p.feat.csv/ && ($Ac=($1-20)/400);
      /LuNG.[Ee][Xx][Pp].*$p.err.csv:\s+(\S+)/ && ($Eu=$1);
      /La.[Ee][Xx][Pp].*$p.err.csv:\s+(\S+)/ && ($Ea=$1);
      /Lb.[Ee][Xx][Pp].*$p.err.csv:\s+(\S+)/ && ($Eb=$1);
      /Lc.[Ee][Xx][Pp].*$p.err.csv:\s+(\S+)/ && ($Ec=$1);
      /LuNG.[Ee][Xx][Pp].*$p: A.*= (\S+),.*= (\S+)/ && ($Du=$1,$Mu=$2);
      /La.[Ee][Xx][Pp].*$p: A.*= (\S+),.*= (\S+)/ && ($Da=$1,$Ma=$2);
      /Lb.[Ee][Xx][Pp].*$p: A.*= (\S+),.*= (\S+)/ && ($Db=$1,$Mb=$2);
      /Lc.[Ee][Xx][Pp].*$p: A.*= (\S+),.*= (\S+)/ && ($Dc=$1,$Mc=$2);
    }
    close FILE;
    printf "%-21s %5.0f %5.0f   %4.2f %4.2f %4.2f %4.2f %4.0f  %4.2f %4.2f %4.2f %4.2f %4.0f  %4.2f %4.2f %4.2f %4.2f %4.0f  %4.2f %4.2f %4.2f %4.2f %4.0f\n", $file,
              ((($Du-1.0)/($Mu+0.1)/($Eu+0.1)/(1-$Au))+(($Da-1.0)/($Ma+0.1)/($Ea+0.1)/(1-$Aa)))/2,
              ((($Db-1.0)/($Mb+0.1)/($Eb+0.1)/(1-$Ab))+(($Dc-1.0)/($Mc+0.1)/($Ec+0.1)/(1-$Ac)))/2,
              $Au,$Eu,$Du,$Mu,(($Du-1.0)/($Mu+0.1)/($Eu+0.1)/(1-$Au)),
              $Aa,$Ea,$Da,$Ma,(($Da-1.0)/($Ma+0.1)/($Ea+0.1)/(1-$Aa)),
              $Ab,$Eb,$Db,$Mb,(($Db-1.0)/($Mb+0.1)/($Eb+0.1)/(1-$Ab)),
              $Ac,$Ec,$Dc,$Mc,(($Dc-1.0)/($Mc+0.1)/($Ec+0.1)/(1-$Ac));
  }
}
