from dataclasses import dataclass
from os.path import join
from typing import List

import numpy as np
import pandas as pd

from evaluating_postprocessing import rec_setup
from utility import configuration


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

    def __init__(self, id, power):
        self.id = id
        self.p_consumed = power


@dataclass
class Generator(Unit):
    p_consumed: float = 0

    def __init__(self, id, power):
        self.id = id
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
    @classmethod
    def build(cls, rec_structure):
        # Create units, users and REC
        rec = cls(id="Renewable Energy Community")
        for user_id, user_setup in rec_setup.items():
            user = User()
            for generator in user_setup['generators']:
                power = pd.read_csv(join("generators", f"{generator}.csv"), sep=';').values
                user.add_unit(Generator(id=generator, power=power))
            for load in user_setup['loads']:
                power = pd.read_csv(join("loads", f"{load}.csv"), sep=';').values
                user.add_unit(Load(id=load, power=power))
            rec.add_unit(user)
        return rec

    def write_out(self):
        # Store results
        results = pd.DataFrame(index=[u.id for u in self.units] + ["rec", ],
                               columns=["e_produced", "e_consumed", "e_self_consumed", "e_injected", "e_withdrawn"])

        for user in self.units:
            results.loc[user.id, :] = np.sum(user.p_produced), np.sum(user.p_consumed), np.sum(
                user.p_self_consumed), np.sum(user.p_injected), np.sum(user.p_withdrawn)

        results.loc["rec", :] = results.sum(axis="rows")
        results.to_csv(join(configuration.config.get("path", "output"), "results.csv"), sep=';')
