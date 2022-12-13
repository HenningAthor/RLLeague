"""
Implements the generation of new bots. This is accomplished through a grammar.
"""

import numpy as np
from typing import Dict

from bot.bot import Bot
from bot.nodes import Node, all_nodes, DecisionNode, LogicParameterNode


def create_bot(bot_id: int, min_depth: int, max_depth: int, env_vars: Dict) -> Bot:
    """
    Creates a bot. After min_depth the probability a leaf will occur rises,
    until it is at 100% at max_depth.

    :param bot_id: Unique id of the bot.
    :param min_depth: Minimal depth of the trees.
    :param max_depth: Maximal depth of the trees.
    :param env_vars: Dict of variables present in the environment.
    :return: New bot.
    """
    bot = Bot(bot_id, f'bot_{bot_id}')
    bot.steering_root = create_tree(min_depth, max_depth, env_vars)
    bot.throttle_root = create_tree(min_depth, max_depth, env_vars)
    bot.jump_root = create_tree(min_depth, max_depth, env_vars)
    bot.boost_root = create_tree(min_depth, max_depth, env_vars)
    bot.handbrake_root = create_tree(min_depth, max_depth, env_vars)
    return bot


def create_tree(min_depth: int, max_depth: int, env_vars: Dict) -> Node:
    """
    Creates a tree after a formal grammar. After min_depth the probability a
    leaf will occur rises, until it is at 100% at max_depth. The root is at
    depth 0. Root will always be a DecisionNode with 'kickoff' as Bool
    variable.

    :param min_depth: Minimal depth of the trees.
    :param max_depth: Maximal depth of the trees.
    :param env_vars: Dict of variables present in the environment.
    :return: Root of the tree.
    """
    kickoff_node = LogicParameterNode()
    kickoff_node.parameter = 'kickoff'

    c1_node = np.random.choice(all_nodes['BRANCH']['ARITHMETIC'])()
    c2_node = np.random.choice(all_nodes['BRANCH']['ARITHMETIC'])()

    c1_node.construct_tree(1, min_depth, max_depth, env_vars)
    c2_node.construct_tree(1, min_depth, max_depth, env_vars)

    root = DecisionNode()
    root.add_child(kickoff_node)
    root.add_child(c1_node)
    root.add_child(c2_node)
    return root
