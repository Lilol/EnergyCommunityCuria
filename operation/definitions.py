from utility.definitions import OrderedEnum


class Status(OrderedEnum):
    OK = 'ok'
    OPTIMAL = 'optimal'
    UNPHYSICAL = 'unphysical'
    ERROR = 'error'
    INVALID = 'invalid'
