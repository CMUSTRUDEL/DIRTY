class Grammar(object):
    def __init__(self, syntax_types, variable_types):
        self.syntax_types = list(sorted(syntax_types))
        self.variable_types = ['<pad>', '<unk>'] + list(sorted(variable_types))

        self.syntax_type_to_id = \
            {type: id for id, type in enumerate(self.syntax_types)}
        self._variable_type_to_id = \
            {type: id for id, type in enumerate(self.variable_types)}

    def variable_type_to_id(self, type_token):
        if type_token in self._variable_type_to_id:
            return self._variable_type_to_id[type_token]

        return self._variable_type_to_id['<unk>']

    @property
    def params(self):
        return self.__dict__

    @classmethod
    def load(cls, params):
        grammar = cls(params['syntax_types'], params['variable_types'])

        return grammar
