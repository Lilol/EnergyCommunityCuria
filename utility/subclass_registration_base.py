class SubclassRegistrationBase:
    _key = None
    _subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls._subclasses) == 0:
            cls._subclasses = {}
        cls._subclasses[cls._key] = cls()

    def __getattr__(self, key):
        return getattr(self._subclasses, key)