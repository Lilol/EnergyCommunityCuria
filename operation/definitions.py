from utility.definitions import OrderedEnum


class Status(OrderedEnum):
    OK = 'ok'
    OPTIMAL = 'optimal'
    UNPHYSICAL = 'unphysical'
    UNFEASIBLE = 'unfeasible'
    UNKNOWN = 'unknown'
    ERROR = 'error'
    INVALID = 'invalid'
