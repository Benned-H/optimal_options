"""This script creates and visualizes the regions used in the OBH example."""

from envs.four_rooms import FourRoomsEnv
from envs.example_regions import create_example_regions


def main():
    """Visualize the example regions, to assist in debugging their creation."""
    env = FourRoomsEnv(render_mode="human")

    _, components = create_example_regions(env)
    env.transition_graphs = components.get_component_subgraphs()
    env.reset()

    input("Press 'enter' to exit.\n")

    env.close()


if __name__ == "__main__":
    main()
