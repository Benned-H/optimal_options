"""Implements the Four Rooms gridworld environment using minigrid.

Original problem adapted from Sutton et al. (1999). Reference:

[1] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A
    framework for temporal abstraction in reinforcement learning,”
    Artificial Intelligence, vol. 112, no. 1, pp. 181–211, Aug. 1999,
    doi: 10.1016/S0004-3702(99)00052-1.
"""

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import SymbolicObsWrapper


class FourRoomsEnv(MiniGridEnv):
    """The Four Rooms MDP environment."""

    def __init__(
        self,
        size: int = 13,
        agent_start_pos: tuple[int, int] = (3, 3),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ):
        """Initialize the Four Rooms environment.

        :param  size                External side length of the grid
        :param  agent_start_pos     Starting (x,y) position of the agent
        :param  agent_start_dir     Starting direction of the agent
        :param  max_steps           Maximum steps in an episode
        """
        # Initialize the starting state of the agent
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:  # Cap the maximum steps
            max_steps = 4 * size**2

        # Initialize the base class, MiniGridEnv
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        """Generate the mission of the agent."""
        return "Reach the green goal."

    def _gen_grid(self, width: int = 13, height: int = 13):
        """Generate the 13x13 four rooms environment.

        Reference: [1] "Between MDPs and semi-MDPs..." by Sutton et al. (1999).

        :param  width       Width (cells) of the grid
        :param  height      Height (cells) of the grid
        """

        # Begin with an empty grid surrounded by four walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Create center vertical walls (x = 6; y = 0 to 2, 4 to 9, 11 to 12)
        self.grid.vert_wall(6, 0, 3)
        self.grid.vert_wall(6, 4, 6)
        self.grid.vert_wall(6, 11, 2)

        # Create left horizontal walls (x = 0 to 1, 3 to 6; y = 6)
        self.grid.horz_wall(0, 6, 2)
        self.grid.horz_wall(3, 6, 4)

        # Create right horizontal walls (x = 6 to 8, 10 to 12; y = 7)
        self.grid.horz_wall(6, 7, 3)
        self.grid.horz_wall(10, 7, 3)

        # Place the goal somewhere in the bottom-right, for now
        # TODO: Randomize goal square placement!
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent where specified, or randomly
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()


def main():
    """Create human-controlled Four Rooms environment for testing."""
    env = FourRoomsEnv(render_mode="human")

    # Use wrapper to make the environment fully observable
    env_obs = SymbolicObsWrapper(env)

    # Enable manual control for testing
    manual_control = ManualControl(env_obs, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
