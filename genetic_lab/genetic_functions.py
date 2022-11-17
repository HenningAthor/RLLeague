import random
import copy
from heapq import nsmallest


def mutate(self, problem, node, strength, depth = 0):
    """
    Mutates nodes by going through all nodes in subtree of the given node.

    :param self: -
    :param problem: ???
    :param node: contains information and functionalities of an individual
    :param strength: probability if node changes
    :param depth: depth in the current subtree starting at the given node
    :return: mutated node
    """
    shouldMutate = random.uniform(0.0, 1.0) <= strength

    if shouldMutate:
        return problem.randomNode(1 - (depth * 0.05))

    else:
        for i in range(node.reqChildren()):
            node.childrenList[i] = self.mutate(problem, node.childrenList[i], strength, depth + 1)

        return node

def s_mutate(stratPar):
    """
    Mutates the given strategy parameter.

    :param stratPar: information about an individual
    :return: mutated strategy parameter
    """
    return stratPar * random.uniform(0.0, 2.0)

def pickRndSubTree(self, node, strength):
    """
    Selects a random node with its subtree.

    :param self: -
    :param node: contains information and functionalities of an individual
    :param strength: the probability for node selection
    :return: a random node with its subtree
    """
    shouldRecombineNode = random.uniform(0.0, 1.0) <= strength

    if not shouldRecombineNode and node.reqChildren() != 0:
        nodeRndChild = random.choice(node.childrenList)

        node = self.pickRndSubTree(nodeRndChild, strength)

    return node

def replaceRndSubTree(self, node, subTree, strength):
    """
    Searches in a tree a random node and replaces this node with its subtree with the given subtree.

    :param self: -
    :param node: contains information and functionalities of an individual
    :param subTree: a node (with subtree in its information)
    :param strength: the probability for node selection
    :return: the given tree or the subtree if node gets replaced
    """
    shouldRecombineNode = random.uniform(0.0, 1.0) <= strength

    if not shouldRecombineNode and node.reqChildren() != 0:
        nodeChildIndex = random.choice(range(len(node.childrenList)))

        nodeRndChild = node.childrenList[nodeChildIndex]

        node.childrenList[nodeChildIndex] = self.replaceRndSubTree(nodeRndChild, subTree, strength)

        return node
    else:
        return subTree

def recombine(self, prog1, prog2, strength1, strength2):
    """
    Changes a random subtree of prog1 with a random subtree of prog2.

    :param self: -
    :param prog1: an individual
    :param prog2: another individual
    :param strength1: probability for node selection
    :param strength2: probability for node selection
    :return: "prog2" with an changed subtree inside
    """
    node1 = self.pickRndSubTree(prog1, strength1)
    node1Copy = copy.deepcopy(node1)
    node2 = self.replaceRndSubTree(prog2, node1Copy, strength2)

    return node2

def s_recombine(selected_parents):
    """
    Recombines the given strategy parameter.

    :param selected_parents: list of individuals
    :return: recombined strategy parameter
    """
    return sum([parent[1] for parent in selected_parents]) / len(selected_parents)

def best_individual(individuals):
    """
    The best individual of current generation.

    :param individuals: list of individuals (trees)
    :return:
    """
    return min(individuals, key=lambda k: k[2])

def marriage(parents, ro):
    """
    :param parents: contains individuals
    :param ro: a number
    :return: "ro" random individuals
    """
    return random.sample(parents, ro)

def selection(offsprings, mu):
    """
    Selection of best individuals from pool of children.

    :param offsprings: list of individuals
    :param mu: a number
    :return: the best programs of "offsprings"
    """
    return nsmallest(mu, offsprings, key=lambda k: k[2])

def plus_selection(offsprings, parents, mu):
    """
    Selection of best individuals from pool of children and parents.

    :param offsprings: list of individuals
    :param parents: list of individuals
    :param mu: a number
    :return: the best programs of "offsprings" and "parents"
    """
    return nsmallest(mu, offsprings + parents, key=lambda k: k[2])

def tournament_selection(offsprings, mu):
    """

    :param offsprings:
    :param mu:
    :return:
    """
    return