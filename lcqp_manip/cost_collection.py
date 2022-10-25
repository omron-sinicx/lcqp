import numpy as np

class CostCollection:
    """ Collection of costs. """
    def __init__(self, objects: list, contacts: list):
        """ Construct a cost collection. 

            Args: 
                objects: List of the objects.
                contacts: List of the contacts.
        """
        self.dimq = 0
        for e in objects:
            self.dimq += e.dimq
        self.dimdf = 5 * len(contacts)
        self.dim = self.dimq + self.dimdf
        self.costs = []

    def append(self, cost):
        """ Append a cost term. """
        self.costs.append(cost)

    def set_opt_vars(self, q, f, v, df):
        """ Sets the optimization variables of the LCQP. """
        for e in self.costs:
            e.set_opt_vars(q, f, v, df)

    def get_qp_cost(self, q_val, f_val):
        """ Gets the QP cost function, i.e., the Hessian and linear term. """
        H = np.zeros([self.dim, self.dim])
        g = np.zeros([self.dim])
        for e in self.costs:
            He, ge = e.get_qp_cost(q_val, f_val)
            H = H + He
            g = g + ge
        return H, g