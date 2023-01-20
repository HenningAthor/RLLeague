"""
Implements the generation of new bots. This is accomplished through a grammar.
"""

from typing import Dict, List

from bot.bot import Bot
from bot.nodes import Node, DecisionNode


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
    bot.creation_variables = env_vars
    return bot


def create_tree(min_depth: int,
                max_depth: int,
                env_vars: Dict[str, List[str]]) -> Node:
    """
    Creates a tree after a formal grammar. After min_depth the probability a
    leaf will occur rises, until it is at 100% at max_depth. The root is at
    depth 0.

    :param min_depth: Minimal depth of the trees.
    :param max_depth: Maximal depth of the trees.
    :param env_vars: Dict of variables present in the environment.
    :return: Root of the tree.
    """
    root = DecisionNode()
    root.construct_tree(1, min_depth, max_depth, env_vars)
    return root
