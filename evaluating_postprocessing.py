import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
class Unit:

    def __init__(self, id):
        self._id = id

    @property
    def id(self):
        return self._id


class Load(Unit):

    def __init__(self, id, consumption):
        super().__init__(id)
        self._consumption = consumption

    @property
    def consumption(self):
        return self._consumption


class Generator(Unit):

    def __init__(self, id, production):
        super().__init__(id)
        self._production = production

    @property
    def production(self):
        return self._production


class User:

    def __init__(self, id, generators=[], loads=[]):

        self._id = id
        self._loads = []
        self._generators = []
        self._evaluated = False

        # Add generators and loads
        self.add_generators(*generators)
        self.add_loads(*loads)

        # Placeholders
        self._p_produced = None
        self._p_consumed = None
        self._p_self_consumed = None
        self._p_injected = None
        self._p_withdrawn = None

    @property
    def id(self):
        return self._id

    @property
    def p_produced(self):
        return self._p_produced

    @property
    def p_consumed(self):
        return self._p_consumed

    @property
    def p_self_consumed(self):
        return self._p_self_consumed

    @property
    def p_injected(self):
        return self._p_injected

    @property
    def p_withdrawn(self):
        return self._p_withdrawn

    @property
    def evaluated(self):
        return self._evaluated

    def add_generators(self, *generators):
        for generator in generators:
            self._add_unit(generator, where="generators")

    def add_loads(self, *loads):
        for load in loads:
            self._add_unit(load, where="loads")

    def _add_unit(self, unit, where):

        if not isinstance(unit, Unit):
            raise TypeError()

        if where not in ["generators", "loads"]:
            raise ValueError()

        if where == "generators":
            self._generators.append(unit)

        else:
            self._loads.append(unit)

        self._evaluated = False

    def evaluate(self):

        # Evaluate total production and consumption
        p_produced = sum([generator.production for generator in self._generators])
        p_consumed = sum([load.consumption for load in self._loads])

        # Calculate self-consumption and net grid exchanges
        p_self_consumed = np.minimum(p_produced, p_consumed)
        p_injected = p_produced - p_self_consumed
        p_withdrawn = p_consumed - p_self_consumed

        # Store values
        self._p_produced = p_produced
        self._p_consumed = p_consumed
        self._p_self_consumed = p_self_consumed
        self._p_injected = p_injected
        self._p_withdrawn = p_withdrawn
        self._evaluated = True


class REC:

    def __init__(self, *users):

        self._users = []
        self._evaluated = False

        # Add users
        self.add_users(*users)

        # Placeholders
        self._p_injected = None
        self._p_withdrawn = None
        self._p_shared = None
        self._p_to_grid = None
        self._p_from_grid = None

    @property
    def p_injected(self):
        return self._p_injected

    @property
    def p_withdrawn(self):
        return self._p_withdrawn

    @property
    def p_shared(self):
        return self._p_shared

    @property
    def p_to_grid(self):
        return self._p_to_grid

    @property
    def p_from_grid(self):
        return self._p_from_grid

    @property
    def evaluated(self):
        return self._evaluated

    def add_users(self, *users):
        for user in users:
            if not isinstance(user, User):
                raise TypeError()
            self._users.append(user)
        self._evaluated = False

    def evaluate(self):

        for user in self._users:
            if not user.evaluated:
                user.evaluate()

        # Evaluate total injections and withdrawals
        p_injected = sum([user.p_injected for user in self._users])
        p_withdrawn = sum([user.p_withdrawn for user in self._users])

        # Evaluate shared energy and net exchanges with national grid
        p_shared = np.minimum(p_injected, p_withdrawn)
        p_to_grid = p_injected - p_shared
        p_from_grid = p_withdrawn - p_shared

        # Store values
        self._p_injected = p_injected
        self._p_withdrawn = p_withdrawn
        self._p_shared = p_shared
        self._p_to_grid = p_to_grid
        self._p_from_grid = p_from_grid
        self._evaluated = True


# ----------------------------------------------------------------------------

# Setup of the REC
rec_setup = {'parrocchia1': {'generators': ['IT001E04397577_20'], 'loads': ['IT001E04397577']},
    'parrocchia2': {'generators': [], 'loads': ['IT001E04170238', 'IT001E04170234', 'IT001E01660522', 'IT001E01658228',
                                                'IT001E00793931']},
    'famiglie': {'generators': [], 'loads': ['famiglie_30']}, }

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
        'e_self_consumed': np.sum(user.p_self_consumed), 'e_injected': np.sum(user.p_injected),
        'e_withdrawn': np.sum(user.p_withdrawn), }, index=[user.id])
    results = pd.concat((results, results_users), axis=0)

results_rec = results.sum().to_frame().T
results_rec['p_shared'] = np.sum(rec.p_shared)
results_rec['p_to_grid'] = np.sum(rec.p_to_grid)
results_rec['p_from_grid'] = np.sum(rec.p_from_grid)
results_rec.index = ['rec']

results = pd.concat((results, results_rec), axis=0)
results.to_csv("results.csv", sep=';')
