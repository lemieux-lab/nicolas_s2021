# Nicolas  (Summer 2021)

This repo contains the code produced for my summer internship (2021) with Lemieux lab at IRIC.

### root files
Files at the repo's root are some of the first codes made for this intership. They are mostly tests and archives of some of my first attempts at Flux, and should not be relied on for anything important.

* flux_testing.jl: Small toy model to test out a basic nn architecture on a trivial problem
* kmer_oracle.jl: First implementation of a pipeline that trains on leucegene data. Only works with a single sample
* kmer_oracle_from_disk.jl: A version of kmer_oracle that works streaming data from the disk. Allows for multiple samples usage. Very slow and no sample traceback.
* kmer_utils.jl: a collection of tools to work with kmers.
* Flux_Conv_Testing.jl: a small toy model to test out basic convolution layer architecture with fake data (doesn't really work. Just wanted to make sure it didn't crash)

### Aboleth directory
Files in the Aboleth directory are more recent and representative of the current pipeline

* Aboleth.jl: Main file for the pipeline. Contains network architecture and general pipeline logic
* disk_utils.jl: Utils to stream data from the disk. Not really used anymore
* kmer_utils.jl: a collection of tools to work with kmers.
* plot_utils.jl: a collection of tools to plot various results from the neural network training
* reindeer_bridge.jl: communicates with the Reindeer API to load the index & query kmers on that index
* run_params.jl: contains the hyperparameters for a given run, aswell as additional infos such as plot paths, what to save, at which frequency, etc...

### Aboleth/reindeer directory
Files in this directory includes the Reindeer source files that have been modified to allow for querying from Julia (Refer to reindeer_bridge.jl for how to operate it)
