# Knowing a Thing by Its Powers

Does the way a transformer ingests an object condition how it generalizes?

A standard transformer keeps a stored row of numbers for each object. Here it is
compared with an otherwise identical model that stores nothing for an object and
computes its vector by reading the interactions in which the object has been
observed. Both predict the outcome, yes or no, of an interaction between two
objects in a synthetic world whose hidden constitution is known.

The idea comes from John Locke, for whom a thing is known through its powers.

- **Paper:** [paper.pdf](paper.pdf), and in Spanish, [paper_es.pdf](paper_es.pdf)
- **Interactive summary:** <https://nousarchives.github.io/blog/angel/knowing-a-thing-by-its-powers/>

## What it found

With enough training data the model that reads is 2 to 4 points more accurate
on interactions it was never shown, and the standard transformer needs about
1.6 times more data to reach the same error.

Its vectors also contain more of the hidden constitution, and the model uses it.
Two results go against it. With little data it overfits and falls behind, and
with the largest training sets a simple factorization with stored rows is more
accurate than both.

## Run it

```bash
pip install torch numpy scipy scikit-learn matplotlib

python exp.py --fase prueba --salida prueba    # smoke test, about a minute on CPU

python exp.py --fase ajuste    --v2 --pasos 20000 --modelos TR,MLP,FAC,OV,TRR --salida results_v2
python exp.py --fase principal --v2 --pasos 20000 --semillas 0,1,2,3,4 --modelos TR,MLP,FAC,OV,TRR --sin_bilineal --salida results_v2
python exp.py --fase largo     --v2 --semillas 0,1,2,3,4 --salida results_v2
python analiza.py --salida results_v2
```

Every finished run is saved and skipped when the command is launched again. The
runs in results_v2/ took about 14 hours on a T4, and analiza.py rebuilds
report.md from them in seconds. results_v1/ holds the first version of the
experiment and results_v2/ the second, the one the paper reports.

Flags, phase names and result keys are in Spanish. ajuste is the tuning phase,
principal the main experiment, largo the long training and prueba a smoke
test; salida is the output folder, semillas the seeds and pasos the steps.

## Limits

A small synthetic world, small models and five seeds. No real data, so nothing here
shows that the design gives better predictions on a real task.

---

Independent work. It follows the format of the final project of MIT 6.7960 Deep
Learning (Fall 2024), on MIT OpenCourseWare, and was not submitted to MIT or
assessed by it. Part of the code was written with the help of Claude.
