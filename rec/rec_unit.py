from dataclasses import dataclass
from os.path import join
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, concat

from utility import configuration


@dataclass
class Unit:
    id: str
    p_produced: float = 0
    p_consumed: float = 0
    p_selfconsumed: float = 0
    p_injected: float = 0
    p_withdrawn: float = 0
    evaluated: bool = False
    to_store = ["p_produced", "p_consumed", "p_selfconsumed", "p_shared", "p_injected", "p_withdrawn"]

    def evaluate(self):
        if self.evaluated:
            return

        # Calculate self-consumption and net grid exchanges
        # Is this lazy eval?
        self.p_selfconsumed = np.minimum(self.p_produced, self.p_consumed)
        self.p_injected = self.p_produced - self.p_selfconsumed
        self.p_withdrawn = self.p_consumed - self.p_selfconsumed
        self.evaluated = True

    def to_df(self):
        return DataFrame(data=[np.sum(getattr(self, data_type)) for data_type in self.to_store], index=[id, ],
                         columns=self.to_store)

    def write_out(self):
        self.to_df().to_csv(join(configuration.config.get("path", "output"), f"{self.id}.csv"), sep=';',
                            float_format="%.4f")


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
    p_shared_with: float = 0
    p_shared_inj: float = 0
    to_store = ["p_produced", "p_consumed", "p_selfconsumed", "p_shared", "p_injected", "p_withdrawn", "p_shared_with",
                "p_shared_inj"]

    def __init__(self, units=None):
        self.units = units

    def add_unit(self, unit):
        self.units.append(unit)

    def evaluate(self):
        for unit in self.units:
            unit.evaluate()

        # Evaluate total production and consumption and selfconsumption
        for quantity in ("p_produced", "p_consumed", "p_injected", "p_withdrawn", "p_selfconsumed"):
            setattr(self, quantity, sum([getattr(unit, quantity) for unit in self.units]))

        super().evaluate()

    def eval_p_shared_with(self, p_with_rec):
        if not self.evaluated:
            self.evaluate()
        self.p_shared_with = self.p_withdrawn / p_with_rec

    def eval_p_shared_inj(self, p_inj_rec):
        if not self.evaluated:
            self.evaluate()
        self.p_shared_inj = self.p_withdrawn / p_inj_rec


@dataclass
class REC(User):
    units: List[User] = None
    p_shared: float = 0

    @classmethod
    def build(cls, rec_structure):
        # Create units, users and REC
        rec = cls(id="rec")
        for user_id, user_setup in rec_structure.items():
            user = User()
            for generator in user_setup['generators']:
                power = pd.read_csv(join("generators", f"{generator}.csv"), sep=';')
                user.add_unit(Generator(id=generator, power=power.values))
            for load in user_setup['loads']:
                power = pd.read_csv(join("loads", f"{load}.csv"), sep=';')
                user.add_unit(Load(id=load, power=power.values))
            rec.add_unit(user)
        return rec

    def evaluate(self):
        super().evaluate()

        for unit in self.units:
            unit.eval_p_shared_inj(self.p_injected)
            unit.eval_p_shared_with(self.p_withdrawn)

    def to_df(self):
        return concat([super().to_df(), *[user.to_df() for user in self.units]])
