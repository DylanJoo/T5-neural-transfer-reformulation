split=$1

# make ntr dataset
cut -f2 $split.history.ntr.tsv > history
cut -f2 $split.rewrite.tsv > rewrite
paste history rewrite > $split.ntr.seq2seq.tsv

# make nqg dataset
cut -f2 $split.history.tsv > history
cut -f2 $split.rewrite.tsv > rewrite
paste history rewrite > $split.nqg.seq2seq.tsv
rm history
rm rewrite
