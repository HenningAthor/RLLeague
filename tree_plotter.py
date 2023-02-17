from agent.nodes import Node
from agent.tree import Tree
from anytree import Node as PlotNode, RenderTree as PlotTree
from anytree.exporter import DotExporter

id = 0

def get_node_representation(x : Node) -> str:
    global id
    id += 1
    return f'{id} {type(x).__name__} P:{x.parameter} C:{x.constant}'

def generate_plot_node(node : Node) -> PlotNode:
    plot_node = PlotNode(get_node_representation(node), value=0)
    plot_node.children = {generate_plot_node(child) for child in node.children}
    return plot_node

def visualize_tree(tree : Tree, out_path : str):
    root = generate_plot_node(tree.root)
    DotExporter(root).to_picture(out_path)

if __name__ == '__main__':
    env_variables = {'ARITHMETIC': ['my_car_x',
                                    'my_car_y',
                                    'my_car_z',
                                    'my_car_velocity_x',
                                    'my_car_velocity_y',
                                    'my_car_velocity_z',
                                    'my_car_rotation_yaw',
                                    'my_car_rotation_pitch',
                                    'my_car_rotation_roll',
                                    'enemy_car_x',
                                    'enemy_car_y',
                                    'enemy_car_z',
                                    'enemy_car_velocity_x',
                                    'enemy_car_velocity_y',
                                    'enemy_car_velocity_z',
                                    'enemy_car_rotation_yaw',
                                    'enemy_car_rotation_pitch',
                                    'enemy_car_rotation_roll',
                                    'ball_x', 'ball_y', 'ball_z',
                                    'ball_velocity_x',
                                    'ball_velocity_y',
                                    'ball_velocity_z',
                                    'ball_rotation_yaw',
                                    'ball_rotation_pitch',
                                    'ball_rotation_roll'],
                     'LOGIC': ['kickoff']}

    visualize_tree(Tree(3, 5, env_variables, -1, 1), 'test_img.png')