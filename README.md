# Bandits Playground
Taking inspiration from Sutton & Barto this repo is a place to experiment with bandits algorithms.

### Included Bandit Algorithms
See `policies.py`
- Random
- Greedy
- Epsilon Greedy
- Softmax
- UCB1


## Requirements
- Python 3.10
  - poetry


## Installation
1. Use Poetry to install dependencies

        poetry install


## Usage
Run the experimenter from `main.py` with command-line arguments

    poetry run main.py {options}
    
For example

    poetry run main.py --nb_bandits=100 --bandit_type=gaussian --steps=500
    
Options

| Flag        | Parameters            | Description                                | Required | Default Value | 
|-------------|-----------------------|--------------------------------------------|----------|---------------|
| nb_bandits  | int                   | The number of bandit arms                  | N        | 10            |
| bandit_type | {bernoulli, gaussian} | How the bandit distributes rewards         | N        | bernoulli     |
| steps       | int                   | How many steps to train for                | N        | 1000          |
| trials      | int                   | How many to times to repeat the experiment | N        | 5             |
