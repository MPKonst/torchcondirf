# TorchCondiRF

A minimal repository with fast and flexible implementations of linear chain conditional random fields (LC-CRFs) in pytorch. 
Its main feature is that it allows one ot easily compute the restricted partition function and top-k Viterbi sequences 
by fixing any sequence positions to any subset of tags.

## Installation
To install the library run `pip install .` in the main `torchcondirf` directory.

The `CrfHead`'s only dependency is `pytorch`. The `StructCrfHead` uses Sasha Rush's `torch-struct` library, so if you'd like to use that, 
you should intall it first: `pip install "git+https://github.com/harvardnlp/pytorch-struct"`.

To install the library in dev mode and run the tests, run `pip install -e '.[dev]' && pytest`.

## Authors and acknowledgment
This implementation was developed by Momchil Konstantinov and Gregorio Benincasa at Eigen Technologies. A lot of initial inspiration was taken from the CRF implementations by AI2 (https://github.com/allenai/allennlp/).

## License
MIT
