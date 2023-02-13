"""
Implements the different nodes.
"""
import random
from typing import List, Dict, Union, Type

import numpy as np
from numba import njit

from agent.util import random_sample


class Node(object):
    def __init__(self):
        # relationship
        self.children: List['Node'] = []
        self.parent: Union['Node', None] = None

        # values needed for bloat elimination
        self.is_bloat: bool = False  # indicates if this is a bloat node
        self.bloat_min: Union[float, bool] = 0.0  # minimum value the node can propagate
        self.bloat_max: Union[float, bool] = 0.0  # maximum value the node can propagate
        self.bloat_val: Union[float, bool] = 0.0  # if the node is bloat, this value will be returned

        self.parameter: str = ''
        self.constant: Union[float, bool] = 0.0

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy function. Deep-copies the children and connects them to this
        node. The parent is not set.

        :param memodict: Dictionary to save already seen variables.
        :return: Deepcopy of the node.
        """
        cls = self.__class__
        new_node = cls.__new__(cls)

        new_node.children = [c.__deepcopy__(memodict) for c in self.children]
        new_node.parent = None  # parent is not set
        for child in new_node.children:
            child.parent = new_node

        new_node.is_bloat = self.is_bloat
        new_node.bloat_min = self.bloat_min
        new_node.bloat_max = self.bloat_max
        new_node.bloat_val = self.bloat_val

        new_node.parameter = self.parameter
        new_node.constant = self.constant

        return new_node

    def add_child(self,
                  child: 'Node') -> None:
        """
        Add the child to the list of children. Creates two-way connection
        between parent and child.

        :param child: A node.
        :return: None
        """
        self.children.append(child)
        child.parent = self

    def remove_child(self,
                     child: 'Node') -> None:
        """
        Remove the child of the list of children. Removes the two-way connection
        between parent and child.

        :param child: A node.
        :return: None
        """
        self.children.remove(child)
        child.parent = None

    def assert_node(self) -> None:
        """
        Function to assert that the node is correct.

        :return: None
        """
        # assert that all children are correct
        for child in self.children:
            child.assert_node()

        # assert number of children correct
        assert all_nodes_info[type(self)]['expected_num_children'] == len(self.children)

        # assert that own variables are correct
        self.assert_self()

        # assert that all connections are correct
        for child in self.children:
            assert child.parent == self

        if self.parent is not None:
            assert self in self.parent.children

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, bool, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Float, Bool or array
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to itself and set its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        raise NotImplementedError

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def unmark_bloat(self) -> None:
        """
        Marks all nodes in the tree as non-bloat, resetting the subtree.

        :return: None
        """
        for child in self.children:
            child.unmark_bloat()

        self.is_bloat = False

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Mutates the node. This can change the type or internal constants.

        :param env_vars: Variables of the environment.
        :return: None
        """
        # possible new types
        new_node_type = random.choice(all_nodes_info[type(self)]['similar_nodes'])

        node = new_node_type()
        node.parent = self.parent
        node.children = self.children

        # manage connections with parent
        swap_nodes(self, node)

        # delete self
        del self

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class ArithmeticNode(Node):
    """
    ArithmeticNode that acts as a superclass. Its children should be ARITHMETIC,
    and it evaluates to an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> [float, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the p that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        n_children = all_nodes_info[type(self)]['expected_num_children']
        for i in range(n_children):
            if random_sample() < p:
                # generate a leaf
                node_type = get_random_node('LEAF', 'ARITHMETIC')
            else:
                # generate a branch
                node_type = get_random_node('BRANCH', 'ARITHMETIC')

            node = node_type()
            node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
            self.add_child(node)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class LogicalNode(Node):
    """
    LogicalNode that acts as a superclass. Its children should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        raise NotImplementedError

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class BinaryLogicalNode(LogicalNode):
    """
    BinaryLogicalNode that acts as a superclass. Its two children should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the p that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        n_children = all_nodes_info[type(self)]['expected_num_children']
        for i in range(n_children):
            if random_sample() < p:
                # generate a leaf
                node_type = get_random_node('LEAF', 'LOGIC')
            else:
                # generate a branch
                node_type = get_random_node('BRANCH', 'LOGIC')

            node = node_type()
            node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
            self.add_child(node)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class UnaryLogicalNode(LogicalNode):
    """
    UnaryLogicalNode that acts as a superclass. Its one child should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the p that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        n_children = all_nodes_info[type(self)]['expected_num_children']
        for i in range(n_children):
            if random_sample() < p:
                # generate a leaf
                node_type = get_random_node('LEAF', 'LOGIC')
            else:
                # generate a branch
                node_type = get_random_node('BRANCH', 'LOGIC')

            node = node_type()
            node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
            self.add_child(node)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class ComparisonNode(LogicalNode):
    """
    ComparisonNode that acts as a superclass. Its children should be ARITHMETIC
    ,and it evaluates to a LOGIC value.
    """

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the p that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        n_children = all_nodes_info[type(self)]['expected_num_children']
        for i in range(n_children):
            if random_sample() < p:
                # generate a leaf
                node_type = get_random_node('LEAF', 'ARITHMETIC')
            else:
                # generate a branch
                node_type = get_random_node('BRANCH', 'ARITHMETIC')

            node = node_type()
            node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
            self.add_child(node)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        raise NotImplementedError

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        raise NotImplementedError


class SumNode(ArithmeticNode):
    """
    SumNode object. Its children should be ARITHMETIC, and it evaluates to an
    ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Sums up the values of its children.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """

        if self.is_bloat:
            return self.bloat_val

        return self.children[0].eval(environment) + self.children[1].eval(environment)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        self.bloat_min = sum(c.bloat_min for c in self.children)
        self.bloat_max = sum(c.bloat_max for c in self.children)

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} + {src_2})'


class ProductNode(ArithmeticNode):
    """
    ProductNode object. Its children should be ARITHMETIC, and it evaluates to
    an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Multiplies all values of its children.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        if self.is_bloat:
            return self.bloat_val

        return self.children[0].eval(environment) * self.children[1].eval(environment)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        a = self.children[0].bloat_min * self.children[1].bloat_min
        b = self.children[0].bloat_min * self.children[1].bloat_max
        c = self.children[0].bloat_max * self.children[1].bloat_min
        d = self.children[0].bloat_max * self.children[1].bloat_max

        self.bloat_min = min(a, b, c, d)
        self.bloat_max = max(a, b, c, d)

        if np.isclose(self.bloat_min, self.bloat_max):
            self.is_bloat = True
            self.bloat_val = self.bloat_min
            self.bloat_max = self.bloat_min

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} * {src_2})'


class AndNode(BinaryLogicalNode):
    """
    AndNode object. Its children should be LOGIC, and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], LogicalNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Applies a logic AND on its children values.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        r1 = self.children[0].eval(environment)
        r2 = self.children[0].eval(environment)

        if (r1 == 1.0 or r1 == 1 or r1 is True) and (r2 == 1.0 or r2 == 1 or r2 is True):
            return True
        return False

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        a = self.children[0].bloat_min & self.children[1].bloat_min
        b = self.children[0].bloat_min & self.children[1].bloat_max
        c = self.children[0].bloat_max & self.children[1].bloat_min
        d = self.children[0].bloat_max & self.children[1].bloat_max

        self.bloat_min = bool(min(a, b, c, d))
        self.bloat_max = bool(max(a, b, c, d))

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} & {src_2})'


class OrNode(BinaryLogicalNode):
    """
    OrNode object. Its children should be LOGIC, and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], LogicalNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Applies a logic OR on its children values.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        r1 = self.children[0].eval(environment)
        r2 = self.children[0].eval(environment)

        if (r1 == 1.0 or r1 == 1 or r1 is True) or (r2 == 1.0 or r2 == 1 or r2 is True):
            return True
        return False

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        a = self.children[0].bloat_min | self.children[1].bloat_min
        b = self.children[0].bloat_min | self.children[1].bloat_max
        c = self.children[0].bloat_max | self.children[1].bloat_min
        d = self.children[0].bloat_max | self.children[1].bloat_max

        self.bloat_min = bool(min(a, b, c, d))
        self.bloat_max = bool(max(a, b, c, d))

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} | {src_2})'


class SmallerNode(ComparisonNode):
    """
    SmallerNode object. Its children should be ARITHMETIC, and it evaluates to a
    LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> [bool, np.ndarray]:
        """
        Return True if the value of the 0th-child is smaller than the value
        of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 < c2
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].bloat_max < self.children[1].bloat_min:
            # child 0 will always be smaller than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.is_bloat = True
            self.bloat_val = True
        elif self.children[0].bloat_min >= self.children[1].bloat_max:
            # child 0 will always be greater-equal than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.is_bloat = True
            self.bloat_val = False
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} < {src_2})'


class SmallerEqualNode(ComparisonNode):
    """
    SmallerEqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> [bool, np.ndarray]:
        """
        Return True if the value of the 0th-child is smaller-equal than the
        value of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 <= c2
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].bloat_max <= self.children[1].bloat_min:
            # child 0 will always be smaller equal than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.is_bloat = True
            self.bloat_val = True
        elif self.children[0].bloat_min > self.children[1].bloat_max:
            # child 0 will always be greater than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.is_bloat = True
            self.bloat_val = False
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} <= {src_2})'


class EqualNode(ComparisonNode):
    """
    EqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Return True if the value of the 0th-child is equal the value of the
        1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 == c2
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        cs = self.children
        # check if ranges dont overlap # TODO: Think about this
        no_overlap = True
        no_overlap &= cs[0].bloat_min > cs[1].bloat_max
        no_overlap &= cs[0].bloat_min < cs[1].bloat_max

        # check if ranges only allow one value
        one_value = True
        one_value &= cs[0].bloat_min == cs[0].bloat_max
        one_value &= cs[1].bloat_min == cs[1].bloat_max
        one_value &= cs[0].bloat_min == cs[1].bloat_min

        if no_overlap:
            # child 0 and child 1 can never be the same
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.is_bloat = True
            self.bloat_val = False
        elif one_value:
            # child 0 and child 1 will always be the same
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.is_bloat = True
            self.bloat_val = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} == {src_2})'


class GreaterEqualNode(ComparisonNode):
    """
    GreaterEqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 >= c2
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].bloat_min >= self.children[1].bloat_max:
            # child 0 will always be greater equal than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.is_bloat = True
            self.bloat_val = True
        elif self.children[0].bloat_max < self.children[1].bloat_min:
            # child 0 will always be smaller than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.is_bloat = True
            self.bloat_val = False
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} >= {src_2})'


class GreaterNode(ComparisonNode):
    """
    GreaterNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 > c2
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].bloat_min > self.children[1].bloat_max:
            # child 0 will always be greater than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.is_bloat = True
            self.bloat_val = True
        elif self.children[0].bloat_max <= self.children[1].bloat_min:
            # child 0 will always be smaller-equal than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.is_bloat = True
            self.bloat_val = False
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        return f'({src_1} > {src_2})'


class NegationNode(UnaryLogicalNode):
    """
    NegationNode object. Its one child should be LOGIC, and it evaluates
    to a LOGIC value. It must have exactly one child.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], LogicalNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Negates the value of its child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        res = self.children[0].eval(environment)
        if isinstance(res, np.ndarray):
            return ~res
        elif isinstance(res, bool):
            return not res
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].is_bloat:
            # the one child is bloat, therefore this node is also bloat
            # only invert the bloat values
            self.is_bloat = True
            if isinstance(self.children[0].bloat_val, np.ndarray):
                self.bloat_val = ~self.children[0].bloat_val
                self.bloat_min = ~self.children[0].bloat_min
                self.bloat_max = ~self.children[0].bloat_max
            else:
                self.bloat_val = not self.children[0].bloat_val
                self.bloat_min = not self.children[0].bloat_min
                self.bloat_max = not self.children[0].bloat_max
        else:
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        return f'(~{src_1})'


class IdentityLogicalNode(UnaryLogicalNode):
    """
    IdentityLogicalNode object. Its one child should be Logic, and it evaluates
    to a LOGIC value. It must have exactly one child.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], LogicalNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Returns the value of its child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        if self.is_bloat:
            return self.bloat_val

        res = self.children[0].eval(environment)
        return res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].is_bloat:
            # the one child is bloat, therefore this node is also bloat
            # just pass the bloat values
            self.is_bloat = True
            self.bloat_val = self.children[0].bloat_val
            self.bloat_min = self.children[0].bloat_min
            self.bloat_max = self.children[0].bloat_max
        else:
            self.bloat_min = False
            self.bloat_max = True

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        return f'({src_1})'


class DecisionNode(ArithmeticNode):
    """
    DecisionNode object. It must have exactly three children. The first child
    should be a LOGIC child, both other children should be ARITHMETIC children.
    """

    def __init__(self):
        super().__init__()

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], ArithmeticNode)
        assert isinstance(self.children[2], ArithmeticNode)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        If the 0th child evaluates to True it will propagate the 1st child
        value, else it will propagate the 2nd child value.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        if self.is_bloat:
            return self.children[self.bloat_val].eval(environment)

        decision = self.children[0].eval(environment)
        res1 = self.children[1].eval(environment)
        res2 = self.children[2].eval(environment)

        res = (decision * res1) + ((1 - decision) * res2)
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the p that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        node_types = ['LOGIC', 'ARITHMETIC', 'ARITHMETIC']
        n_children = all_nodes_info[type(self)]['expected_num_children']
        for i in range(n_children):
            if random_sample() < p:
                # generate a leaf
                node_type = get_random_node('LEAF', node_types[i])
            else:
                # generate a branch
                node_type = get_random_node('BRANCH', node_types[i])

            node = node_type()
            node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
            self.add_child(node)

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for child in self.children:
            child.determine_bloat(env_stats)

        self.bloat_min = min(self.children[1].bloat_min, self.children[2].bloat_min)
        self.bloat_max = max(self.children[1].bloat_max, self.children[2].bloat_max)

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Changes the ordering of the ARITHMETIC children.

        :param env_vars: Variables of the environment.
        :return: None
        """
        self.children[1], self.children[2] = self.children[2], self.children[1]

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        src_1 = self.children[0].numba_jit(env_variables, headers)
        src_2 = self.children[1].numba_jit(env_variables, headers)
        src_3 = self.children[2].numba_jit(env_variables, headers)
        return f'({src_1} * {src_2} + (1 - {src_1}) * {src_3})'


class ArithmeticConstantNode(ArithmeticNode):
    """
    ConstantNode object. Any children will be ignored, and it will only return
    one constant value.
    """

    def __init__(self):
        super().__init__()

        self.constant = 1.0

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        pass

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> float:
        """
        Returns the float constant.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        res = self.constant
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the constant value of the node between [0, 1).

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = random_sample()

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        self.bloat_min = self.constant
        self.bloat_max = self.constant

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Changes the constant.

        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = np.random.random_sample()

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        return f'({self.constant})'


class LogicalConstantNode(LogicalNode):
    """
    LogicalConstantNode object. Any children will be ignored, and it will only
    return one constant value.
    """

    def __init__(self):
        super().__init__()

        self.constant = True

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        pass

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> bool:
        """
        Returns the boolean constant.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        res = self.constant
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the constant value of the node between [0, 1).

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = random_sample() < 0.5

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        self.bloat_min = self.constant
        self.bloat_max = self.constant

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Changes the constant.

        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = random_sample() < 0.5

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        return f'({int(self.constant)})'


class ArithmeticParameterNode(ArithmeticNode):
    """
    ArithmeticParameterNode object. Any children will be ignored, and it will
    only return the parameter of the environment, which is an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()

        self.parameter = 'NONE'

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        pass

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Returns the float value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        res = environment['ARITHMETIC'][self.parameter]
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the parameter for the node.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = random.choice(env_vars['ARITHMETIC'])

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        self.bloat_min = env_stats['ARITHMETIC'][self.parameter]['min']
        self.bloat_max = env_stats['ARITHMETIC'][self.parameter]['max']

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Changes the observed variable.

        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = random.choice(env_vars['ARITHMETIC'])

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        idx = headers.index(self.parameter)
        return f"arr[:, {idx}]"


class LogicParameterNode(LogicalNode):
    """
    LogicParameterNode object. Any children will be ignored, and it will only
    return the parameter of the environment, which is a LOGIC value.
    """

    def __init__(self):
        super().__init__()

        self.parameter = 'NONE'

    def assert_self(self):
        """
        Asserts all variables, which are specific to the node.

        :return: None
        """
        pass

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[bool, np.ndarray]:
        """
        Returns the boolean value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        :return: Bool.
        """
        res = environment['LOGIC'][self.parameter]
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the parameter for the node.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = random.choice(env_vars['LOGIC'])

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines the bloat values for this node.
        It will first determine the bloat values for its children and then the
        bloat values for itself.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        self.bloat_min = env_stats['LOGIC'][self.parameter]['min']
        self.bloat_max = env_stats['LOGIC'][self.parameter]['max']

    def mutate(self,
               env_vars: Dict[str, List[str]]) -> None:
        """
        Changes the observed variable.

        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = random.choice(env_vars['LOGIC'])

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> str:
        """
        Writes python code in a string, so it can be compiled.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: Python code as a string.
        """
        idx = headers.index(self.parameter)
        return f"arr[:, {idx}].astype(np.int64)"


# available types, nodes, probabilities and information
all_types = ['ARITHMETIC', 'LOGIC', 'DECISION', 'NONE']
all_branch_nodes = np.array([SumNode, DecisionNode, AndNode, OrNode, SmallerNode, SmallerEqualNode,
                             GreaterEqualNode, GreaterNode, EqualNode,
                             NegationNode, IdentityLogicalNode])
all_nodes = {
    'BRANCH': {'ARITHMETIC': np.array([SumNode, DecisionNode]),
               'LOGIC': np.array([AndNode, OrNode, SmallerNode, GreaterNode, NegationNode, IdentityLogicalNode])
               },
    'LEAF': {'ARITHMETIC': np.array([ArithmeticConstantNode, ArithmeticParameterNode]),
             'LOGIC': np.array([LogicalConstantNode, LogicParameterNode])
             }
}
all_nodes_p = {
    'BRANCH': {'ARITHMETIC': np.array([0.8, 0.2]),
               'LOGIC': np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
               },
    'LEAF': {'ARITHMETIC': np.array([0.1, 0.9]),
             'LOGIC': np.array([0.1, 0.9])
             }
}

all_nodes_info = {Node: {'type': 'UNINITIALIZED', 'children_type': 'UNINITIALIZED', 'similar_nodes': [], 'expected_num_children': 0},
                  ArithmeticNode: {'type': 'ARITHMETIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SumNode, ProductNode], 'expected_num_children': 2},
                  LogicalNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [], 'expected_num_children': 2},
                  BinaryLogicalNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [AndNode, OrNode], 'expected_num_children': 2},
                  UnaryLogicalNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [NegationNode, IdentityLogicalNode], 'expected_num_children': 1},
                  ComparisonNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SmallerNode, SmallerEqualNode, EqualNode, GreaterEqualNode, GreaterNode], 'expected_num_children': 2},
                  SumNode: {'type': 'ARITHMETIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [ProductNode], 'expected_num_children': 2},
                  ProductNode: {'type': 'ARITHMETIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SumNode], 'expected_num_children': 2},
                  AndNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [OrNode], 'expected_num_children': 2},
                  OrNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [AndNode], 'expected_num_children': 2},
                  SmallerNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [GreaterNode], 'expected_num_children': 2},
                  # SmallerEqualNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SmallerNode, EqualNode, GreaterEqualNode, GreaterNode], 'expected_num_children': 2},
                  # EqualNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SmallerNode, SmallerEqualNode, GreaterEqualNode, GreaterNode], 'expected_num_children': 2},
                  # GreaterEqualNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SmallerNode, SmallerEqualNode, EqualNode, GreaterNode], 'expected_num_children': 2},
                  GreaterNode: {'type': 'LOGIC', 'children_type': 'ARITHMETIC', 'similar_nodes': [SmallerNode], 'expected_num_children': 2},
                  NegationNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [IdentityLogicalNode], 'expected_num_children': 1},
                  IdentityLogicalNode: {'type': 'LOGIC', 'children_type': 'LOGIC', 'similar_nodes': [NegationNode], 'expected_num_children': 1},
                  DecisionNode: {'type': 'ARITHMETIC', 'children_type': 'DECISION', 'similar_nodes': [], 'expected_num_children': 3},
                  ArithmeticConstantNode: {'type': 'ARITHMETIC', 'children_type': 'NONE', 'similar_nodes': [], 'expected_num_children': 0},
                  LogicalConstantNode: {'type': 'LOGICAL', 'children_type': 'NONE', 'similar_nodes': [], 'expected_num_children': 0},
                  ArithmeticParameterNode: {'type': 'ARITHMETIC', 'children_type': 'NONE', 'similar_nodes': [], 'expected_num_children': 0},
                  LogicParameterNode: {'type': 'LOGICAL', 'children_type': 'NONE', 'similar_nodes': [], 'expected_num_children': 0}
                  }

"""
Operations on nodes.
"""

# for faster random node generation
random_nodes = {'BRANCH': {'ARITHMETIC': [], 'LOGIC': []},
                'LEAF': {'ARITHMETIC': [], 'LOGIC': []}}


def get_random_node(branch_type: str,
                    node_type: str) -> Type['Node']:
    """
    Function to quickly get a random node from all nodes.

    :param branch_type: Either 'BRANCH' or 'LEAF'
    :param node_type: Either 'ARITHMETIC' or 'LOGIC'
    :return: A node type
    """
    # if our list is empty, generate new samples
    if not random_nodes[branch_type][node_type]:
        random_nodes[branch_type][node_type] = random.choices(all_nodes[branch_type][node_type], weights=all_nodes_p[branch_type][node_type], k=10000)

    return random_nodes[branch_type][node_type].pop()


def swap_nodes(node_1: 'Node',
               node_2: 'Node',
               swap_children: bool = True) -> None:
    """
    Swaps the first node with the second node. Also swaps the two-way
    connection between the children and parents.

    :param node_1: The first node.
    :param node_2: The second node.
    :param swap_children: If children should also be swapped.
    :return: None
    """
    # get the parents
    node_1_parent = node_1.parent
    node_2_parent = node_2.parent

    if node_1_parent is not None:
        # node 1 has a parent, manage connections
        idx = node_1_parent.children.index(node_1)
        node_1_parent.children[idx] = node_2

    node_2.parent = node_1_parent

    if node_2_parent is not None:
        # node 2 has a parent, manage connections
        idx = node_2_parent.children.index(node_2)
        node_2_parent.children[idx] = node_1

    node_1.parent = node_2_parent

    if swap_children:
        node_1_children = node_1.children
        node_2_children = node_2.children

        for child in node_1_children:
            child.parent = node_2

        for child in node_2_children:
            child.parent = node_1

        node_1.children = node_2_children
        node_2.children = node_1_children


def recombine_nodes(node_1: Node, node_2: Node) -> None:
    """
    Node 1 and Node 2 are in different trees. This function will swap both nodes
    in their respective tree. It will also connect the parents.
    Only nodes with the same type can be swapped.

    :param node_1: Node 1 in tree 1.
    :param node_2: Node 2 in tree 2.
    :return: None
    """
    swap_nodes(node_1, node_2, swap_children=False)
