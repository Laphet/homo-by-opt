bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e tmp.err -o tmp.out "python test_ex1.py"