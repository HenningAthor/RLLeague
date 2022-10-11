"""
Implements the different nodes.
"""
import math
import numpy as np
from typing import List, Dict


class Node(object):
    def __init__(self):
        # properties
        self.type: str = "UNINITIALIZED"
        self.children_type: str = "UNINITIALIZED"
        self.immutable: bool = False

        # relationship
        self.num_children: int = 0
        self.children: List['Node'] = []
        self.parent: 'Node' = None

        # values needed for bloat elimination
        self.bloat: bool = False  # indicates if this is a bloat node
        self.bloat_min = float('inf')  # minimum value the node can propagate
        self.bloat_max = -float('inf')  # maximum value the node can propagate

    def add_child(self, child: 'Node'):
        """
        Add the child to the list of children and sets the parent of the child.
        Increments the number of children.

        :param child: Another root.
        :return: None
        """
        self.children.append(child)
        child.parent = self
        self.num_children += 1

    def remove_child(self, child: 'Node'):
        """
        Remove the child of the list of children and removes the parent of the
        child. Decrements the number of children.

        :param child: Another root.
        :return: None
        """
        child.parent = None
        self.children.remove(child)
        self.num_children -= 1

    def eval(self, environment):
        """
        Evaluates the root.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: None
        """
        raise NotImplementedError

    def construct_tree(self,
                       depth: int,
                       min_depth: int,
                       max_depth: int,
                       env_vars: Dict) -> 'Node':
        """
        Constructs a tree with this node as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree, counted from the first node,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
        :return: None
        """
        raise NotImplementedError

    def determine_bloat(self):
        """
        Determines the bloat values for this node.

        :return: None
        """
        raise NotImplementedError


class ArithmeticNode(Node):
    """
    ArithmeticNode that acts as a superclass. Its children should be ARITHMETIC
    and it evaluates to an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Evaluates the root.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: None
        """
        raise NotImplementedError

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
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

    def determine_bloat(self):
        """
        Determines the bloat values for this node.

        :return: None
        """
        raise NotImplementedError


class LogicalNode(Node):
    """
    LogicalNode that acts as a superclass. Its children should be LOGIC
    and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Evaluates the root.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: None
        """
        raise NotImplementedError

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
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

    def determine_bloat(self):
        """
        Determines the bloat values for this node.

        :return: None
        """
        raise NotImplementedError


class ComparisonNode(Node):
    """
    ComparisonNode that acts as a superclass. Its children should be ARITHMETIC
    and it evaluates to a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Evaluates the root.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: None
        """
        raise NotImplementedError

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
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

    def determine_bloat(self):
        """
        Determines the bloat values for this node.

        :return: None
        """
        raise NotImplementedError


class SumNode(ArithmeticNode):
    """
    SumNode object. Its children should be ARITHMETIC and it evaluates to an
    ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Sums up the values of its children.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: Sum
        """
        res = 0
        for child in self.children:
            res += child.eval(environment)
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Sum up min values and max
        values from the children.

        :return: None
        """
        self.bloat_min = sum(c.bloat_min for c in self.children)
        self.bloat_max = sum(c.bloat_max for c in self.children)


class ProductNode(ArithmeticNode):
    """
    ProductNode object. Its children should be ARITHMETIC and it evaluates to
    an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Multiplies all values of its children.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: Product
        """
        res = 1
        for child in self.children:
            res *= child.eval(environment)
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Sum up min values and max
        values from the children.

        :return: None
        """
        # TODO: Think about this!
        self.bloat_min = math.prod(c.bloat_min for c in self.children)
        self.bloat_max = math.prod(c.bloat_max for c in self.children)


class AndNode(LogicalNode):
    """
    AndNode object. Its children should be LOGIC and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"

    def eval(self, environment):
        """
        Applies a logic AND on its children values.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: Logical AND
        """
        res = True
        for child in self.children:
            res = res and child.eval(environment)
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        self.bloat_min = True
        self.bloat_max = True
        for child in self.children:
            self.bloat_min = self.bloat_min and child.bloat_min
            self.bloat_max = self.bloat_max and child.bloat_max


class OrNode(LogicalNode):
    """
    OrNode object. Its children should be LOGIC and it evaluates to a LOGIC
    value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 2

        self.bloat_min = 0
        self.bloat_max = 1

    def eval(self, environment):
        """
        Applies a logic OR on its children values.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: Logical OR
        """
        res = True
        for child in self.children:
            res = res or child.eval(environment)
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        self.bloat_min = False
        self.bloat_max = False
        for child in self.children:
            self.bloat_min = self.bloat_min or child.bloat_min
            self.bloat_max = self.bloat_max or child.bloat_max


class SmallerNode(ComparisonNode):
    """
    SmallerNode object. Its children should be ARITHMETIC and it evaluates to a
    LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Return True if the value of the 0th-child is smaller than the value
        of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 < c2
        """
        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 < c2
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        if self.children[0].bloat_max < self.children[1].bloat_min:
            # child 0 will always be smaller than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.bloat = True
        elif self.children[0].bloat_min >= self.children[1].bloat_max:
            # child 0 will always be greater-equal than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True


class SmallerEqualNode(ComparisonNode):
    """
    SmallerEqualNode object. Its children should be ARITHMETIC and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Return True if the value of the 0th-child is smaller-equal than the
        value of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 <= c2
        """
        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 <= c2
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        if self.children[0].bloat_max <= self.children[1].bloat_min:
            # child 0 will always be smaller equal than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.bloat = True
        elif self.children[0].bloat_min > self.children[1].bloat_max:
            # child 0 will always be greater than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True


class GreaterEqualNode(ComparisonNode):
    """
    GreaterEqualNode object. Its children should be ARITHMETIC and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 >= c2
        """
        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 >= c2
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        if self.children[0].bloat_min >= self.children[1].bloat_max:
            # child 0 will always be greater equal than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.bloat = True
        elif self.children[0].bloat_max < self.children[1].bloat_min:
            # child 0 will always be smaller than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True


class GreaterNode(ComparisonNode):
    """
    GreaterNode object. Its children should be ARITHMETIC and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Return True if the value of the 0th-child is greater-equal the value
        of the 1st-child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 > c2
        """
        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 > c2
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        if self.children[0].bloat_min > self.children[1].bloat_max:
            # child 0 will always be greater than child 1
            # node will always evaluate to True
            self.bloat_min = True
            self.bloat_max = True
            self.bloat = True
        elif self.children[0].bloat_max <= self.children[1].bloat_min:
            # child 0 will always be smaller-equal than child 1
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True


class EqualNode(ComparisonNode):
    """
    EqualNode object. Its children should be ARITHMETIC and it evaluates
    to a LOGIC value. It must have exactly two children.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "ARITHMETIC"
        self.num_children = 2

    def eval(self, environment):
        """
        Return True if the value of the 0th-child is equal the value of the
        1st-child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 == c2
        """
        c1 = self.children[0].eval(environment)
        c2 = self.children[1].eval(environment)
        res = c1 == c2
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        # check if ranges dont overlap
        no_overlap = True
        no_overlap = no_overlap and self.children[0].bloat_min > self.children[
            1].bloat_max
        no_overlap = no_overlap and self.children[0].bloat_min < self.children[
            1].bloat_max

        # check if ranges only allow one value
        one_value = True
        one_value = one_value and self.children[0].bloat_min == self.children[
            0].bloat_max
        one_value = one_value and self.children[1].bloat_min == self.children[
            1].bloat_max
        one_value = one_value and self.children[0].bloat_min == self.children[
            1].bloat_min

        if no_overlap:
            # child 0 and child 1 can never be the same
            # node will always evaluate to False
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        elif one_value:
            # child 0 and child 1 will always be the same
            # node will always evaluate to True
            self.bloat_min = False
            self.bloat_max = False
            self.bloat = True
        else:
            # node ranges overlap, so no conclusion can be made
            self.bloat_min = False
            self.bloat_max = True


class NegationNode(LogicalNode):
    """
    NegationNode object. Its child should be Logic and it evaluates
    to a LOGIC value. It must have exactly one child.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "LOGIC"
        self.num_children = 1

    def eval(self, environment):
        """
        Negates the value of its child.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: Not c1
        """
        res = not self.children[0].eval(environment)
        return res

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        # Reverse the bloat values
        self.bloat_min = not self.children[0].bloat_min
        self.bloat_max = not self.children[0].bloat_max
        # get bloat state from child
        self.bloat = self.children[0].bloat


class DecisionNode(Node):
    """
    DecisionNode object. It must have exactly three children. The first child
    should be a LOGIC child, both other children should be ARITHMETIC children.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "DECISION"
        self.num_children = 3

    def eval(self, environment):
        """
        If the 0th child evaluates to True it will propagate the 1st child
        value, else it will propagate the 2nd child value.

        :param environment: Dictionary holding values for parameters.
        Will be passed on to the children.
        :return: c1 or c2, depending on c0.
        """
        if self.children[0].eval(environment):
            res = self.children[1].eval(environment)
        else:
            res = self.children[2].eval(environment)
        return res

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Constructs a tree with this root as root. It will generate children
        and append them to its self and sets its corresponding values. After
        min_depth the chance of leaves rises to 100% at max_depth.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
        :return: None
        """
        # determine the probability that a leaf root occurs
        incr = 1.0 / (max_depth - min_depth)
        p = 0.0 if depth < min_depth else incr * (depth - min_depth)

        # decide for every root if it will be a leaf
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

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        # Reverse the bloat values
        same = self.children[0].bloat_min == self.children[0].bloat_max
        same_true = same and self.children[0].bloat_min is True
        same_false = same and self.children[0].bloat_min is False

        if same_true:
            # logic node will only evaluate to True
            # propagate bloat values from child 1
            self.bloat_min = self.children[1].bloat_min
            self.bloat_max = self.children[1].bloat_max
            self.children[2].bloat = True
        elif same_false:
            # logic node will only evaluate to False
            # propagate bloat values from child 2
            self.bloat_min = self.children[2].bloat_min
            self.bloat_max = self.children[2].bloat_max
            self.children[1].bloat = True


class ConstantNode(Node):
    """
    ConstantNode object. Any children will be ignored and it will only return
    one constant value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "NONE"

        self.constant = 0.0

    def eval(self, environment):
        """
        Returns the constant.

        :param environment: Dictionary holding values for parameters (unused).
        :return: Constant.
        """
        res = self.constant
        return res

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Randomly sets the constant value of the root between [0, 1).

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
        :return: None
        """
        self.constant = np.random.random_sample()

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat values for this node
        are 0 and 1.

        :return: None
        """
        # Reverse the bloat values
        self.bloat_min = self.constant
        self.bloat_max = self.constant


class ArithmeticParameterNode(Node):
    """
    ArithmeticParameterNode object. Any children will be ignored and it will
    only return the parameter of the environment, which is an ARITHMETIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "ARITHMETIC"
        self.children_type = "NONE"

        self.parameter = 0

    def eval(self, environment):
        """
        Returns the value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        Will be used to extract the value.
        :return: Arithmetic value.
        """
        res = environment[self.type][self.parameter]
        return res

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Randomly sets the parameter for the root.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
        :return: None
        """
        self.parameter = np.random.choice(env_vars['ARITHMETIC'])

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat can not be computed by
        this function.

        :return: None
        """
        # bloat can not be calculated by this function currently!
        pass


class LogicParameterNode(Node):
    """
    LogicParameterNode object. Any children will be ignored and it will only
    return the parameter of the environment, which is a LOGIC value.
    """

    def __init__(self):
        super().__init__()
        self.type = "LOGIC"
        self.children_type = "NONE"

        self.parameter = 0

    def eval(self, environment):
        """
        Returns the value for the set parameter.

        :param environment: Dictionary holding values for parameters.
        Will be used to extract the value.
        :return: Logic value.
        """
        res = environment[self.type][self.parameter]
        return res

    def construct_tree(self, depth: int, min_depth: int, max_depth: int,
                       env_vars: Dict):
        """
        Randomly sets the parameter for the root.

        :param depth: Current depth of the tree, counted from the first root,
        which called this function.
        :param min_depth: Minimal depth the tree should have.
        :param max_depth: Maximum depth the tree should have.
        :param env_vars: Dictionary holding the variables, which will be
        present in the environment.
        :return: None
        """
        self.parameter = np.random.choice(env_vars['LOGIC'])

    def determine_bloat(self):
        """
        Determines the bloat values for this node. Bloat can not be computed by
        this function.

        :return: None
        """
        # bloat can not be calculated by this function currently!
        pass


# available types, nodes and probabilities
all_types = ['ARITHMETIC', 'LOGIC', 'DECISION', 'NONE']
all_nodes = {
    'BRANCH': {'ARITHMETIC': [SumNode, DecisionNode],
               'LOGIC': [AndNode, OrNode, SmallerNode, SmallerEqualNode,
                         GreaterEqualNode, GreaterNode, EqualNode,
                         NegationNode]
               },
    'LEAF': {'ARITHMETIC': [ConstantNode, ArithmeticParameterNode],
             'LOGIC': [LogicParameterNode]
             }
}
all_nodes_p = {
    'BRANCH': {'ARITHMETIC': [0.8, 0.2],
               'LOGIC': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                         0.125]
               },
    'LEAF': {'ARITHMETIC': [0.1, 0.9],
             'LOGIC': [1.0]
             }
}

"""
Operations on nodes.
"""


def count_nodes(root: Node):
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


def count_non_bloat_nodes(root: Node):
    """
    Recursively counts the number of non bloat nodes.

    :param root: Root node.
    :return: Total number of nodes.
    """
    n = 0
    node_list = [root]
    while node_list:
        node = node_list.pop()
        if not node.bloat:
            # only track nodes, which are not bloat
            n += 1
            node_list.extend(node.children)
    return n


def leaf_type_count(root: Node):
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
            # node has no child, so its a leaf
            if type(node) in par_count:
                par_count[type(node)] += 1
            else:
                par_count[type(node)] = 1
        else:
            node_list.extend(node.children)
    return par_count


def bloat_analysis(root: Node, env_stats: Dict):
    """
    Applies a Min-Max-Analysis on the tree with the specified root. Uses the
    statistics applied in env_stats. Will mark nodes with no information
    value as bloat.

    :param root: Root node.
    :param env_stats: Dictionary holding statistics for the parameters of the
    environment.
    :return: None
    """
    # set bloat_min and bloat_max in the leaf nodes
    node_list = [root]
    next_parent_list = []
    parent_list = []

    while node_list:
        node = node_list.pop()
        if node.children_type == 'NONE':
            # save this node for later
            if node not in parent_list:
                parent_list.append(node)

            # set for leaf nodes
            if isinstance(node, ConstantNode):
                # min and max are the same for ConstantNode
                node.bloat_min = node.constant
                node.bloat_max = node.constant
            elif isinstance(node, ArithmeticParameterNode):
                # set min and max after the environment statistics
                node.bloat_min = env_stats['ARITHMETIC'][node.parameter]['min']
                node.bloat_max = env_stats['ARITHMETIC'][node.parameter]['max']
            elif isinstance(node, LogicParameterNode):
                # set min and max after the environment statistics
                node.bloat_min = env_stats['LOGIC'][node.parameter]['min']
                node.bloat_max = env_stats['LOGIC'][node.parameter]['max']

    # propagate the min and max values up
    # stop when parent list has the roots parent
    while parent_list != [None]:

        # empty the parent list
        while parent_list:
            node = parent_list.pop()

            # determine the bloat
            node.determine_bloat()

            # append parent if not already present
            if node not in next_parent_list:
                next_parent_list.append(node.parent)

            parent_list = next_parent_list
            next_parent_list = []

    # propagate the bloat mark down the tree
    node_list = [root]
    while node_list:
        node = node_list.pop()
        for child in node.children:
            if child.bloat:
                # mark the whole subtree of child as bloat
                mark_bloat_subtree(child)
            else:
                # append the child to investigate its children
                node_list.append(child)


def mark_bloat_subtree(root: Node):
    """
    Marks every node in the tree with this root as bloat.

    :param root: Root node.
    :return: None
    """
    node_list = [root]
    while node_list:
        node = node_list.pop()
        for child in node.children:
            child.bloat = True
            node_list.append(child)
