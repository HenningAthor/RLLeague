"""
Implements the different nodes.
"""
import random
from typing import List, Dict, Union

import numpy as np


class Node(object):
    def __init__(self):
        # properties
        self.type: str = "UNINITIALIZED"
        self.children_type: str = "UNINITIALIZED"
        self.immutable: bool = False
        self.similar_node_types: List = []  # holds all similar Node types

        # relationship
        self.num_children: int = 0
        self.children: List['Node'] = []
        self.parent: Union['Node', None] = None

        # values needed for bloat elimination
        self.is_bloat: bool = False  # indicates if this is a bloat node
        self.bloat_min: Union[float, bool, None] = None  # minimum value the node can propagate
        self.bloat_max: Union[float, bool, None] = None  # maximum value the node can propagate
        self.bloat_val: Union[int, float, bool] = 0.0  # if the node is bloat, this value will be returned

        self.last_eval = None

    def add_child(self,
                  child: 'Node') -> None:
        """
        Add the child to the list of children.
        Increments the number of children.

        :param child: Another node.
        :return: None
        """
        self.children.append(child)

    def remove_child(self,
                     child: 'Node') -> None:
        """
        Remove the child of the list of children.
        Decrements the number of children.

        :param child: Another root.
        :return: None
        """
        self.children.remove(child)

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> Union[float, bool]:
        """
        Evaluates the node.

        :param environment: Dictionary holding values for parameters.
        :return: Float or Bool
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
        Marks all nodes in the tree as not bloat, resetting the subtree.

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
        node_types = self.similar_node_types.copy()
        node_types.remove(self.__class__)

        new_node_type = random.choice(node_types)

        node = new_node_type()
        node.parent = self.parent
        node.children = self.children

        # manage connections with parent
        if self.parent is not None:
            self.parent.add_child(node)
            self.parent.remove_child(self)

        # manage connections with children
        for child in self.children:
            child.parent = node

        # delete self
        del self


class ArithmeticNode(Node):
    """
    ArithmeticNode that acts as a superclass. Its children should be ARITHMETIC,
    and it evaluates to an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2
        self.similar_node_types = [SumNode, ProductNode]

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
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
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the probability that a leaf root occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every root if it will be a leaf
        for i in range(self.num_children):
            if np.random.random_sample() < p:
                # generate a leaf
                nodes = all_nodes['LEAF']['ARITHMETIC']
                nodes_p = all_nodes_p['LEAF']['ARITHMETIC']
                node_type = np.random.choice(nodes, p=nodes_p)
                node = node_type()
                node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
                self.add_child(node)
            else:
                # generate a branch
                nodes = all_nodes['BRANCH']['ARITHMETIC']
                nodes_p = all_nodes_p['BRANCH']['ARITHMETIC']
                node_type = np.random.choice(nodes, p=nodes_p)
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


class LogicalNode(Node):
    """
    LogicalNode that acts as a superclass. Its children should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
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
        Constructs a tree with this root as root. It will generate children
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


class BinaryLogicalNode(LogicalNode):
    """
    BinaryLogicalNode that acts as a superclass. Its two children should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 2
        self.similar_node_types = [AndNode, OrNode]

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
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
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the probability that a leaf root occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every root if it will be a leaf
        for i in range(self.num_children):
            if np.random.random_sample() < p:
                # generate a leaf
                nodes = all_nodes['LEAF']['LOGIC']
                nodes_p = all_nodes_p['LEAF']['LOGIC']
                node_type = np.random.choice(nodes, p=nodes_p)
                node = node_type()
                node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
                self.add_child(node)
            else:
                # generate a branch
                nodes = all_nodes['BRANCH']['LOGIC']
                nodes_p = all_nodes_p['BRANCH']['LOGIC']
                node_type = np.random.choice(nodes, p=nodes_p)
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


class UnaryLogicalNode(LogicalNode):
    """
    UnaryLogicalNode that acts as a superclass. Its one child should be
    LOGIC, and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 1
        self.similar_node_types = [NegationNode, IdentityLogicalNode]

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
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
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the probability that a leaf root occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every root if it will be a leaf
        for i in range(self.num_children):
            if np.random.random_sample() < p:
                # generate a leaf
                nodes = all_nodes['LEAF']['LOGIC']
                nodes_p = all_nodes_p['LEAF']['LOGIC']
                node_type = np.random.choice(nodes, p=nodes_p)
                node = node_type()
                node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
                self.add_child(node)
            else:
                # generate a branch
                nodes = all_nodes['BRANCH']['LOGIC']
                nodes_p = all_nodes_p['BRANCH']['LOGIC']
                node_type = np.random.choice(nodes, p=nodes_p)
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


class ComparisonNode(LogicalNode):
    """
    ComparisonNode that acts as a superclass. Its children should be ARITHMETIC
    ,and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2
        self.similar_node_types = [SmallerNode, SmallerEqualNode, EqualNode, GreaterEqualNode, GreaterNode]

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
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
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the probability that a leaf root occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every root if it will be a leaf
        for i in range(self.num_children):
            if np.random.random_sample() < p:
                # generate a leaf
                nodes = all_nodes['LEAF']['ARITHMETIC']
                nodes_p = all_nodes_p['LEAF']['ARITHMETIC']
                node_type = np.random.choice(nodes, p=nodes_p)
                node = node_type()
                node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
                self.add_child(node)
            else:
                # generate a branch
                nodes = all_nodes['BRANCH']['ARITHMETIC']
                nodes_p = all_nodes_p['BRANCH']['ARITHMETIC']
                node_type = np.random.choice(nodes, p=nodes_p)
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


class SumNode(ArithmeticNode):
    """
    SumNode object. Its children should be ARITHMETIC, and it evaluates to an
    ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Sums up the values of its children.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

        if self.is_bloat:
            return self.bloat_val

        res = 0.0
        for child in self.children:
            res += child.eval(environment)
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

        self.bloat_min = sum(c.bloat_min for c in self.children)
        self.bloat_max = sum(c.bloat_max for c in self.children)

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min


class ProductNode(ArithmeticNode):
    """
    ProductNode object. Its children should be ARITHMETIC, and it evaluates to
    an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Multiplies all values of its children.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

        if self.is_bloat:
            return self.bloat_val

        res = 1.0
        for child in self.children:
            res *= child.eval(environment)
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


class AndNode(BinaryLogicalNode):
    """
    AndNode object. Its children should be LOGIC, and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Applies a logic AND on its children values.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], LogicalNode)

        if self.is_bloat:
            return self.bloat_val

        res = True
        for child in self.children:
            res = res & child.eval(environment)
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

        a = self.children[0].bloat_min & self.children[1].bloat_min
        b = self.children[0].bloat_min & self.children[1].bloat_max
        c = self.children[0].bloat_max & self.children[1].bloat_min
        d = self.children[0].bloat_max & self.children[1].bloat_max

        self.bloat_min = bool(min(a, b, c, d))
        self.bloat_max = bool(max(a, b, c, d))

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min


class OrNode(BinaryLogicalNode):
    """
    OrNode object. Its children should be LOGIC, and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 2

        self.bloat_min = 0
        self.bloat_max = 1

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Applies a logic OR on its children values.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], LogicalNode)

        if self.is_bloat:
            return self.bloat_val

        res = False
        for child in self.children:
            res = res | child.eval(environment)
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

        a = self.children[0].bloat_min | self.children[1].bloat_min
        b = self.children[0].bloat_min | self.children[1].bloat_max
        c = self.children[0].bloat_max | self.children[1].bloat_min
        d = self.children[0].bloat_max | self.children[1].bloat_max

        self.bloat_min = bool(min(a, b, c, d))
        self.bloat_max = bool(max(a, b, c, d))

        if self.bloat_min == self.bloat_max:
            self.is_bloat = True
            self.bloat_val = self.bloat_min


class SmallerNode(ComparisonNode):
    """
    SmallerNode object. Its children should be ARITHMETIC, and it evaluates to a
    LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Return True if the value of the 0th-child is smaller than the value
        of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

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


class SmallerEqualNode(ComparisonNode):
    """
    SmallerEqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Return True if the value of the 0th-child is smaller-equal than the
        value of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

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


class GreaterEqualNode(ComparisonNode):
    """
    GreaterEqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

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


class GreaterNode(ComparisonNode):
    """
    GreaterNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

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


class EqualNode(ComparisonNode):
    """
    EqualNode object. Its children should be ARITHMETIC, and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Return True if the value of the 0th-child is equal the value of the
        1st-child, False otherwise.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 2
        assert isinstance(self.children[0], ArithmeticNode)
        assert isinstance(self.children[1], ArithmeticNode)

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


class NegationNode(UnaryLogicalNode):
    """
    NegationNode object. Its one child should be LOGIC, and it evaluates
    to a LOGIC value. It must have exactly one child.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 1

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Negates the value of its child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 1
        assert isinstance(self.children[0], LogicalNode)

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
        assert self.num_children == len(self.children) == 1
        assert isinstance(self.children[0], LogicalNode)

        for child in self.children:
            child.determine_bloat(env_stats)

        if self.children[0].is_bloat:
            # the one child is bloat, therefore this node is also bloat
            # only invert the bloat values
            self.is_bloat = True
            if isinstance(self.children[0].bloat_val, np.ndarray):
                self.bloat_val = ~self.children[0].bloat_val
            elif isinstance(self.children[0].bloat_val, bool):
                self.bloat_val = not self.children[0].bloat_val

            if isinstance(self.children[0].bloat_min, np.ndarray):
                self.bloat_min = ~self.children[0].bloat_min
            elif isinstance(self.children[0].bloat_min, bool):
                self.bloat_min = not self.children[0].bloat_min

            if isinstance(self.children[0].bloat_max, np.ndarray):
                self.bloat_max = ~self.children[0].bloat_max
            elif isinstance(self.children[0].bloat_max, bool):
                self.bloat_max = not self.children[0].bloat_max

        else:
            self.bloat_min = False
            self.bloat_max = True


class IdentityLogicalNode(UnaryLogicalNode):
    """
    IdentityLogicalNode object. Its one child should be Logic, and it evaluates
    to a LOGIC value. It must have exactly one child.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 1

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Returns the value of its child.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 1
        assert isinstance(self.children[0], LogicalNode)

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


class DecisionNode(ArithmeticNode):
    """
    DecisionNode object. It must have exactly three children. The first child
    should be a LOGIC child, both other children should be ARITHMETIC children.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "DECISION"
        self.num_children = 3

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        If the 0th child evaluates to True it will propagate the 1st child
        value, else it will propagate the 2nd child value.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        assert self.num_children == len(self.children) == 3
        assert isinstance(self.children[0], LogicalNode)
        assert isinstance(self.children[1], ArithmeticNode)
        assert isinstance(self.children[2], ArithmeticNode)

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
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        # determine the probability that a leaf node occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every node if it will be a leaf
        node_types = ['LOGIC', 'ARITHMETIC', 'ARITHMETIC']
        for i in range(3):
            if np.random.random_sample() < p:
                # generate a leaf
                nodes = all_nodes['LEAF'][node_types[i]]
                nodes_p = all_nodes_p['LEAF'][node_types[i]]
                node_type = np.random.choice(nodes, p=nodes_p)
                node = node_type()
                node.construct_tree(depth + 1, min_depth, max_depth, env_vars)
                self.add_child(node)
            else:
                # generate a branch
                nodes = all_nodes['BRANCH'][node_types[i]]
                nodes_p = all_nodes_p['BRANCH'][node_types[i]]
                node_type = np.random.choice(nodes, p=nodes_p)
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


class ConstantArithmeticNode(ArithmeticNode):
    """
    ConstantNode object. Any children will be ignored, and it will only return
    one constant value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "NONE"
        self.num_children = 0

        self.constant = 1.0

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Returns the float constant.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        assert self.num_children == len(self.children) == 0

        res = self.constant
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the constant value of the root between [0, 1).

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = np.random.random_sample()

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


class ConstantLogicalNode(LogicalNode):
    """
    ConstantLogicalNode object. Any children will be ignored, and it will only
    return one constant value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGICAL"
        self.children_type = "NONE"
        self.num_children = 0

        self.constant = True

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Returns the boolean constant.

        :param environment: Dictionary holding values for parameters.
        :return: Bool
        """
        assert self.num_children == len(self.children) == 0

        res = self.constant
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the constant value of the root between [0, 1).

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.constant = bool(random.getrandbits(1))

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
        self.constant = bool(random.getrandbits(1))


class ArithmeticParameterNode(ArithmeticNode):
    """
    ArithmeticParameterNode object. Any children will be ignored, and it will
    only return the parameter of the environment, which is an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "NONE"
        self.num_children = 0

        self.parameter = 'NONE'

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Returns the float value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        :return: Float
        """
        assert self.num_children == len(self.children) == 0

        res = environment[self.type][self.parameter]
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the parameter for the root.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = np.random.choice(env_vars['ARITHMETIC'])

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
        self.parameter = np.random.choice(env_vars['ARITHMETIC'])


class LogicParameterNode(LogicalNode):
    """
    LogicParameterNode object. Any children will be ignored, and it will only
    return the parameter of the environment, which is a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "NONE"
        self.num_children = 0

        self.parameter = 'NONE'

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]]) -> bool:
        """
        Returns the boolean value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        :return: Bool.
        """
        assert self.num_children == len(self.children) == 0

        res = environment[self.type][self.parameter]
        return res

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict[str, List[str]]) -> None:
        """
        Randomly sets the parameter for the root.

        :param depth: Current depth of the tree.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Variables of the environment.
        :return: None
        """
        self.parameter = np.random.choice(env_vars['LOGIC'])

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
        self.parameter = np.random.choice(env_vars['LOGIC'])


# available types, nodes and probabilities
all_types = ['ARITHMETIC', 'LOGIC', 'DECISION', 'NONE']
all_nodes = {
    'BRANCH': {'ARITHMETIC': [SumNode, ProductNode, DecisionNode],
               'LOGIC': [AndNode, OrNode, SmallerNode, SmallerEqualNode,
                         GreaterEqualNode, GreaterNode, EqualNode,
                         NegationNode, IdentityLogicalNode]
               },
    'LEAF': {'ARITHMETIC': [ConstantArithmeticNode, ArithmeticParameterNode],
             'LOGIC': [ConstantLogicalNode, LogicParameterNode]
             }
}
all_nodes_p = {
    'BRANCH': {'ARITHMETIC': [0.4, 0.4, 0.2],
               'LOGIC': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                         0.125, 0.0]
               },
    'LEAF': {'ARITHMETIC': [0.1, 0.9],
             'LOGIC': [0.1, 0.9]
             }
}

"""
Operations on nodes.
"""


def count_nodes(root: Node) -> int:
    """
    Recursively counts the number of nodes.

    :param root: Root node.
    :return: Total number of nodes.
    """
    n = 0
    node_list = [root]
    while node_list:
        node = node_list.pop()
        n += 1
        node_list.extend(node.children)
    return n


def count_non_bloat_nodes(root: Node) -> int:
    """
    Recursively counts the number of non bloat nodes.

    :param root: Root node.
    :return: Total number of nodes.
    """
    n = 0
    node_list = [root]
    while node_list:
        node = node_list.pop()
        if not node.is_bloat:
            n += 1
        for child in node.children:
            if not child.is_bloat:
                node_list.append(child)
    return n


def leaf_type_count(root: Node) -> Dict[str, int]:
    """
    Counts the number of leaf_types with respect to the constant/parameter.

    :param root: Root node.
    :return: Dictionary holding the count for each parameter.
    """
    par_count = {}
    node_list = [root]
    while node_list:
        node = node_list.pop()
        if node.children_type == 'NONE':
            # node has no child, so it's a leaf
            if node.__class__.__name__ not in par_count:
                par_count[node.__class__.__name__] = 0
            par_count[node.__class__.__name__] += 1
        else:
            node_list.extend(node.children)
    return par_count
