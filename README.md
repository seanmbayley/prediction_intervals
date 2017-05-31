# Meta_tune

Meta_tune is an open source Python package for tuning and meta-tuning Random Forests for the purpose of constructing informative prediction intervals.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python3
* git


### Setup

Clone the repository.

```
git clone https://github.com/smbayley/meta_tune.git
```

Cd into the cloned directory and install requirements.

```
cd meta_tune/
pip3 install -r requirements.txt
```

Pip will begin installing all necessary requirements. Once completed, you will be ready to run meta_tune. Learning an RF can be computationally expensive, especially when the dataset is sufficiently large. In order to facilitate configurable interval analysis, we split the tool into two modules: simulate and intervals.  

## Example on Apache-Ant
We have included a sample data directory in the repository (meta_tune/data). This directory includes data for 5 releases of the Apache-Ant open-source project. More information about the data can be found [here](http://openscience.us/repo/defect/ck/ant.html). 

### Simulation
Cd into the src directory.

```
cd meta_tune/src
```

Run main.py in simulation mode.

```
python3 main.py simulate -d ../data/raw/ant -s "sim.Standard -meta_splits 0.25 0.50 0.75" -t all
```
The tool will perform configuration, tuning, and meta-tuning simulation and store the processed data in an automatically created a directory: meta_tune/data/processed/ant/<today's date>/Standard_66-34/. 

Apache Ant is a rather large project. Depending upon your system's resources, the duration of the simulation could be on the order of days. 

### Interval Analysis
Move the simulated data into a directory that's easier to access.

```
mv meta_tune/data/processed/ant/<today's date>/Standard_66-34/* meta_tune/data/processed/ant
```

Perform the tuning benefit operation.

```
python3 main.py intervals -m "core.PIManager -d ../data/processed/ant -ncs 90 95 99" -op "core.Tuning" -out ant_tuning
```

Perform the meta-tuning benefit operation

```
python3 main.py intervals -m "core.PIManager -d ../data/processed/ant -ncs 90 95 99" -op "core.MetaTuning -meta_splits 25 50 75" -out ant_meta
```

The operations will create two CSV files (ant_tuning.csv and ant_meta.csv) in an automatically created directory: meta_tune/data/results.
