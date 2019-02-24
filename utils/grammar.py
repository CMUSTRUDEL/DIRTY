class Grammar(object):
    def __init__(self, syntax_types, variable_types):
        self.syntax_types = list(sorted(syntax_types))
        self.variable_types = list(sorted(variable_types))

        self.syntax_type2id = {type: id for id, type in enumerate(self.syntax_types)}
        self.variable_type2id = {type: id for id, type in enumerate(self.variable_types)}
