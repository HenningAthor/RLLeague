import random
from typing import Dict, List, Union, Type

import numpy as np
from numba import njit

from agent.nodes import DecisionNode, Node, all_branch_nodes, recombine_nodes
from agent.util import scale_continuous, scale_discrete, random_sample


class Tree(object):

    def __init__(self,
                 min_depth: int,
                 max_depth: int,
                 env_vars: Dict[str, List[str]],
                 return_min: float,
                 return_max: float,
                 discrete_return: bool = True):
        self.root: Node = DecisionNode()
        self.creation_variables: Dict[str, List[str]] = env_vars

        self.return_min: float = return_min
        self.return_max: float = return_max
        self.discrete_return: bool = discrete_return
        self.post_process_func: callable = scale_continuous

        self.numba_jit_function: Union[callable, None] = None
        self.is_jit_npy: bool = False

        self.set_eval_post_process()

        if not min_depth == max_depth == -1:
            self.root.construct_tree(1, min_depth, max_depth, env_vars)

    def __deepcopy__(self,
                     memodict={}) -> 'Tree':
        """
        Deep-copies the tree.

        :param memodict: Dictionary to save already seen variables.
        :return: Deepcopy of the tree.
        """
        new_tree = Tree(-1, -1, {}, 0, 0, False)
        new_tree.root = self.root.__deepcopy__(memodict)
        new_tree.creation_variables = self.creation_variables
        new_tree.return_min = self.return_min
        new_tree.return_max = self.return_max
        new_tree.discrete_return = self.discrete_return
        new_tree.post_process_func = self.post_process_func
        new_tree.is_jit_npy = self.is_jit_npy
        new_tree.numba_jit_function = self.numba_jit_function

        return new_tree

    def assert_tree(self):
        """
        Asserts that the tree is correct.

        :return: None
        """
        self.root.assert_node()

    def set_discrete_return(self,
                            discrete_return: bool) -> None:
        """
        Sets a new discrete_return value and also updates the post-processing
        function.

        :return: None
        """
        self.discrete_return = discrete_return
        self.set_eval_post_process()

    def set_eval_post_process(self) -> None:
        """
        Sets the eval_post_process function based on discrete_return value.

        :return: None
        """
        if self.discrete_return:
            self.post_process_func = scale_discrete
        else:
            self.post_process_func = scale_continuous

    def eval(self,
             environment: Dict[str, Dict[str, Union[float, bool]]],
             arr: np.ndarray = None) -> Union[float, np.ndarray]:
        """
        Evaluates the tree.

        :param environment: Dictionary holding values for parameters.
        :param arr: (Optional) The original array holding the data.
        :return: Float or array
        """
        if self.is_jit_npy and arr is not None:
            res = self.numba_jit_function(arr)
        else:
            res = self.root.eval(environment)

        x_min = self.root.bloat_min
        x_max = self.root.bloat_max

        if x_min == x_max:
            return x_min

        post_res = self.post_process_func(res, x_min, x_max, self.return_min, self.return_max)
        return post_res

    def eval_multiprocessing(self,
                             idx: int,
                             return_dict,
                             environment: Dict[str, Dict[str, Union[float, bool]]],
                             arr: np.ndarray = None) -> None:
        """
        Evaluates the tree.

        :param idx: Index of the process.
        :param return_dict: Dictionary to return values.
        :param environment: Dictionary holding values for parameters.
        :param arr: (Optional) The original array holding the data.
        :return: Float or array
        """
        if self.is_jit_npy and arr is not None:
            res = self.numba_jit_function(arr)
        else:
            res = self.root.eval(environment)

        x_min = self.root.bloat_min
        x_max = self.root.bloat_max

        if x_min == x_max:
            x_min = 0.0
            x_max = 1.0

        post_res = self.post_process_func(res, x_min, x_max, self.return_min, self.return_max)
        return_dict[idx] = post_res

    def determine_bloat(self,
                        env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Determines all bloat nodes of this tree. Also calculates the minimum and
        maximum possible value of each node.

        :return: None
        """
        self.root.determine_bloat(env_stats)

    def unmark_bloat(self) -> None:
        """
        Marks all nodes of the tree as non-bloat.

        :return: None
        """
        self.is_jit_npy = False
        self.root.unmark_bloat()

    def mutate(self,
               p: float):
        """
        Mutates the tree. This operation is done in-place.

        :param p: Probability that a node will change.
        :return: None
        """
        work_list = []
        work_list.extend(self.root.children)

        while work_list:
            node = work_list.pop()

            if random_sample() < p:
                # mutate the node
                node.mutate(self.creation_variables)
                self.is_jit_npy = False

            work_list.extend(node.children)

    def numba_jit(self,
                  env_variables: Dict[str, List[str]],
                  headers: List[str]) -> None:
        """
        Creates python compiled code to be executed instead of the eval
        function of the tree.

        :param env_variables: Variables in the environment.
        :param headers: Name of the columns.
        :return: None
        """
        if self.is_jit_npy:
            return

        # generate source code
        src_code = self.root.numba_jit(env_variables, headers)
        src_code = f'import numpy as np\nfrom numba import njit\n\n@njit\ndef npy_jit_eval(arr):\n\treturn {src_code}'
        # Converting above source code to an executable
        local_variables = {}
        code = compile(src_code, '<string>', 'exec')

        # Running the executable code, this loads npy_jit_eval into pythons global runtime... i think...
        exec(code, globals(), local_variables)

        self.is_jit_npy = True
        self.numba_jit_function = local_variables['npy_jit_eval']

    def get_all_nodes(self,
                      node_type: Type[Node]) -> List[Node]:
        """
        Gets all nodes in the subtree, with the specified node type.

        :param node_type: The type of node.
        :return: List of all found nodes.
        """
        nodes = []
        work_list = []
        work_list.extend(self.root.children)
        while work_list:
            node = work_list.pop()

            if type(node) == node_type:
                nodes.append(node)

            work_list.extend(node.children)

        return nodes

    def count_nodes(self) -> int:
        """
        Counts the number of nodes in this tree.

        :return: Number of nodes.
        """
        n = 0
        work_list = [self.root]

        while work_list:
            node = work_list.pop()
            n += 1
            work_list.extend(node.children)

        return n

    def count_non_bloat_nodes(self) -> int:
        """
        Counts the number of nodes in this tree.

        :return: Number of nodes.
        """
        n = 0
        work_list = [self.root]

        while work_list:
            node = work_list.pop()
            if not node.is_bloat:
                n += 1
            work_list.extend(node.children)

        return n


def recombine_trees(tree_1: 'Tree',
                    tree_2: 'Tree') -> None:
    """
    Recombines the two trees. Will swap two subtrees with one another.
    This operation is in-place.

    :param tree_1: The first tree.
    :param tree_2: The second tree.
    """
    tree_1.is_jit_npy = False
    tree_2.is_jit_npy = False

    # choose uniformly which node type we will swap
    node_type = random.choice(all_branch_nodes)

    # get all candidates
    tree_1_nodes = tree_1.get_all_nodes(node_type)
    tree_2_nodes = tree_2.get_all_nodes(node_type)

    # if both trees have candidates
    if tree_1_nodes and tree_2_nodes:
        # choose two nodes uniformly
        node_1 = random.choice(tree_1_nodes)
        node_2 = random.choice(tree_2_nodes)

        assert type(node_1) == type(node_2) == node_type

        recombine_nodes(node_1, node_2)
