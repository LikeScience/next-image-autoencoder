from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.core.constants import OBJECT_TO_IDX
import random


class Env(MiniGridEnv):
    def __init__(
        self,
        wmap,
        map_dims: list,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        colors=None, 
        seed=7342,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_seed = seed
        self.world_map = wmap
        self.exploring=False
        self.colors=colors

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=map_dims[0],
            height=map_dims[1],
            max_steps=256,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "mission"
    
    def _gen_grid(self, width, height):
      self.grid = Grid(width, height)
      self.grid.wall_rect(0, 0, width, height)

      self.valid_actions = {Actions.left, Actions.right, Actions.forward}
      if self.colors is not None:
          this_colors = self.colors
      else:
          this_colors = [[5 for _ in range(width)] for _ in range(height)]
      for i in range(1,height-1):
          for j in range(1,width-1):
            #   if (i,j) != self.agent_start_pos:
            obj = WorldObj.decode(self.world_map[i][j],this_colors[i][j],1) #Assuming all objects with same color and doors start closed
            if obj is not None:
                self.grid.set(j,i,obj) 
      if self.agent_start_pos is not None and not self.exploring:
          self.agent_pos = self.agent_start_pos
          self.agent_dir = self.agent_start_dir
      else:
          self.place_agent_random_pos()
    
    def place_agent_random_pos(self):
        self.agent_pos = [0,0]
        while(self.grid.get(self.agent_pos[0],self.agent_pos[1]) is not None and self.grid.get(self.agent_pos[0],self.agent_pos[1]).type == 'wall'):
            self.agent_pos = (random.randint(1,self.width-2),random.randint(1,self.height-2))
        self.agent_dir = random.randint(0,3)

    def get_array_repr(self):
        grid_array = self.unwrapped.grid.encode()[:,:,0]
        grid_array[self.agent_pos[0],self.agent_pos[1]]=OBJECT_TO_IDX['agent']
        return grid_array.T
        
    def get_colors_repr(self):
        grid_array = self.unwrapped.grid.encode()[:,:,1]
        return grid_array.T
    

