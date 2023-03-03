from agent.nodes import Node
from agent.tree import Tree
from anytree import Node as PlotNode, RenderTree as PlotTree
from anytree.exporter import DotExporter

from recorded_data.data_util import load_min_max_csv, generate_env_stats

ID = 0


def get_node_representation(x: Node) -> str:
    global ID
    ID += 1
    return f'{ID} {type(x).__name__}\nP:{x.parameter}\nC:{x.constant}\n{x.bloat_min}\n{x.bloat_max}'


def generate_plot_node(node: Node) -> PlotNode:
    plot_node = PlotNode(get_node_representation(node), value=0)
    plot_node.children = {generate_plot_node(child) for child in node.children}
    return plot_node


def visualize_tree(tree: Tree, out_path: str):
    root = generate_plot_node(tree.root)
    DotExporter(root).to_picture(out_path)


if __name__ == '__main__':
    env_variables = {'ARITHMETIC': ['ball/pos_x',
                                    'ball/pos_y',
                                    'ball/pos_z',
                                    'ball/vel_x',
                                    'ball/vel_y',
                                    'ball/vel_z',
                                    'ball/ang_vel_x',
                                    'ball/ang_vel_y',
                                    'ball/ang_vel_z',
                                    'player1/pos_x',
                                    'player1/pos_y',
                                    'player1/pos_z',
                                    'player1/vel_x',
                                    'player1/vel_y',
                                    'player1/vel_z',
                                    'player1/ang_vel_x',
                                    'player1/ang_vel_y',
                                    'player1/ang_vel_z',
                                    'player1/boost_amount',
                                    'inverted_player2/pos_x',
                                    'inverted_player2/pos_y',
                                    'inverted_player2/pos_z',
                                    'inverted_player2/vel_x',
                                    'inverted_player2/vel_y',
                                    'inverted_player2/vel_z',
                                    'inverted_player2/ang_vel_x',
                                    'inverted_player2/ang_vel_y',
                                    'inverted_player2/ang_vel_z',
                                    'player2/boost_amount'],
                     'LOGIC': ['player1/on_ground',
                               'player1/ball_touched',
                               'player1/has_jump',
                               'player1/has_flip',
                               'player2/on_ground',
                               'player2/ball_touched',
                               'player2/has_jump',
                               'player2/has_flip']}

    min_max_data, min_max_headers = load_min_max_csv()
    env_stats = generate_env_stats(env_variables, min_max_data, min_max_headers)

    t = Tree(3, 5, env_variables, -1, 1)
    t.determine_bloat(env_stats)
    visualize_tree(t, 'test_img.png')
