import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import configuration


@dataclass
class Unit:
    id: str
    p_produced: float = 0
    p_consumed: float = 0
    p_self_consumed: float = 0
    p_injected: float = 0
    p_withdrawn: float = 0
    evaluated: bool = False

    def evaluate(self):
        if self.evaluated:
            return

        # Calculate self-consumption and net grid exchanges
        # Is this lazy eval?
        self.p_self_consumed = np.minimum(self.p_produced, self.p_consumed)
        self.p_injected = self.p_produced - self.p_self_consumed
        self.p_withdrawn = self.p_consumed - self.p_self_consumed
        self.evaluated = True


@dataclass
class Load(Unit):
    p_produced: float = 0

    def __init__(self, power):
        self.p_consumed = power


@dataclass
class Generator(Unit):
    p_consumed: float = 0

    def __init__(self, power):
        self.p_produced = power


@dataclass
class User(Unit):
    units: List[Unit] = None

    def __init__(self, units=None):
        self.units = units

    def add_unit(self, unit):
        self.units.append(unit)

    def evaluate(self):
        for unit in self.units:
            unit.evaluate()

        # Evaluate total production and consumption
        self.p_produced = sum([unit.p_produced for unit in self.units])
        self.p_consumed = sum([unit.p_consumed for unit in self.units])

        super().evaluate()


@dataclass
class REC(User):
    pass

# ----------------------------------------------------------------------------

# Setup of the REC
# TODO: into input file
# TODO: think about a good input file structure for this (table works fine)
rec_setup = json.loads(configuration.config.get("rec", "setup_file"))

# Create units, users and REC
users = []
for user, user_setup in rec_setup.items():
    generators = []
    loads = []
    for generator in user_setup['generators']:
        power = pd.read_csv(f"Generators//{generator}.csv", sep=';').values
        generators.append(Generator(id=generator, production=power))
    for load in user_setup['loads']:
        power = pd.read_csv(f"Loads//{load}.csv", sep=';').values
        loads.append(Load(id=load, consumption=power))
    users.append(User(user, generators=generators, loads=loads))
rec = REC(*users)

# Evaluate REC
rec.evaluate()

# Store results
results = pd.DataFrame()

for user in rec._users:
    results_users = pd.DataFrame({'e_produced': np.sum(user.p_produced), 'e_consumed': np.sum(user.p_consumed),
                                  'e_self_consumed': np.sum(user.p_self_consumed),
                                  'e_injected': np.sum(user.p_injected), 'e_withdrawn': np.sum(user.p_withdrawn), },
                                 index=[user.id])
    results = pd.concat((results, results_users), axis=0)

results_rec = results.sum().to_frame().T
results_rec['p_shared'] = np.sum(rec.p_shared)
results_rec['p_to_grid'] = np.sum(rec.p_to_grid)
results_rec['p_from_grid'] = np.sum(rec.p_from_grid)
results_rec.index = ['rec']

results = pd.concat((results, results_rec), axis=0)
results.to_csv("results.csv", sep=';')
