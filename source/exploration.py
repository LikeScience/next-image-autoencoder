import numpy as np
from source import agents
from tqdm import tqdm

color_dict = {0: 0, 5: 0.1, 2:-0.1, 3: 0.2, 4:-0.2}

def add_color_bias(arr,env,observation_space):
    if observation_space == "allocentric":
        color_repr = env.get_colors_repr().flatten()
    elif observation_space == "egocentric":
        color_repr = env.gen_obs()['image'][:,:,1].flatten()
    arr_new= arr.astype(np.float32) + np.array([color_dict.get(i, 0) for i in color_repr])
    return arr_new

def exploration_egocentric(env,agent_type=["random",[0.333333,0.333333,0.333334],73],n_steps_train=8000,n_steps_test=2000,n_restart_train=0,n_restart_test=0,rgb_rep=False,with_colors=False):
    
    env.exploring=True
    if (agent_type[0]=="random"):
        probs,seed = agent_type[1:]
        action_agent = agents.random_egocentric_agent(seed,probs)

    image_list_train = []
    pos_list_train = []
    dir_list_train = []

    if (n_restart_train > 0):
        restart_train = set(range(n_steps_train//n_restart_train,n_steps_train,n_steps_train//n_restart_train))
    else:
        restart_train = []

    for i in tqdm(range(n_steps_train)):
        action = action_agent.act()
        if rgb_rep:
            arr = env.unwrapped.get_pov_render(tile_size=1)
        else:
            arr = env.gen_obs()['image'][:,:,0].flatten()
            if with_colors:
                arr = add_color_bias(arr,env,"egocentric")
        arr = np.append(arr, action) 
        image_list_train.append(arr)
        pos_list_train.append((env.agent_pos[0],env.agent_pos[1]))
        dir_list_train.append(env.agent_dir)
        env.step(action) #Step is only applied here
        if (i in restart_train):
            env.reset()
            env.place_agent_random_pos()

    if (n_restart_test > 0):
        restart_test = set(range(n_steps_test//n_restart_test,n_steps_test,n_steps_test//n_restart_test))
    else:
        restart_test = []

    env.reset()
    image_list_test = []
    pos_list_test = []
    dir_list_test = []

    for i in tqdm(range(n_steps_test)):
        action = action_agent.act()
        if rgb_rep:
            arr = env.unwrapped.get_pov_render(tile_size=1)
        else:
            arr = env.gen_obs()['image'][:,:,0].flatten()
            if with_colors:
                arr = add_color_bias(arr,env,"egocentric")
        arr = np.append(arr, action)  
        image_list_test.append(arr)
        pos_list_test.append((env.agent_pos[0],env.agent_pos[1]))
        dir_list_test.append(env.agent_dir)
        env.step(action)
        if (i in restart_test):
            env.reset()
            env.place_agent_random_pos()

    return image_list_train, pos_list_train, dir_list_train, image_list_test, pos_list_test, dir_list_test

def exploration_allocentric(env,agent_type=["random",[0.25,0.25,0.25,0.25],73],n_steps_train=8000,n_steps_test=2000,n_restart_train=0,n_restart_test=0,rgb_rep=False,with_colors=False):
    if (agent_type[0]=="random"):
        probs,seed = agent_type[1:]
        action_agent = agents.random_allocentric_agent(seed,probs)

    env.exploring=True
    image_list_train = []
    pos_list_train = []
    dir_list_train = []

    if (n_restart_train > 0):
        restart_train = set(range(n_steps_train//n_restart_train,n_steps_train,n_steps_train//n_restart_train))
    else:
        restart_train = []

    for i in tqdm(range(n_steps_train)):
        action = action_agent.act(env.agent_dir)
        if rgb_rep:
            env.unwrapped.tile_size = 1
            arr = env.render().flatten()
        else:
            arr = env.get_array_repr().flatten()
            if with_colors:
                arr = add_color_bias(arr,env,"allocentric")
        pos_list_train.append((env.agent_pos[0],env.agent_pos[1]))
        for j in action:
            env.step(j)
        arr = np.append(arr, env.agent_dir)
        dir_list_train.append(env.agent_dir)
        image_list_train.append(arr)
        if (i in restart_train):
            env.reset(seed=env.env_seed)
            env.place_agent_random_pos()

    if (n_restart_test > 0):
        restart_test = set(range(n_steps_test//n_restart_test,n_steps_test,n_steps_test//n_restart_test))
    else:
        restart_test = []

    env.reset()
    image_list_test = []
    pos_list_test = []
    dir_list_test = []

    for i in tqdm(range(n_steps_test)):
        action = action_agent.act(env.agent_dir)
        if rgb_rep:
            env.unwrapped.tile_size = 1
            arr = env.render().flatten()
        else:
            arr = env.get_array_repr().flatten()
            if with_colors:
                arr = add_color_bias(arr,env,"allocentric")
        pos_list_test.append((env.agent_pos[0],env.agent_pos[1]))
        for j in action:
            env.step(j)
        arr = np.append(arr, env.agent_dir) 
        dir_list_test.append(env.agent_dir)
        image_list_test.append(arr)
        if (i in restart_test):
            env.reset(seed=env.env_seed)
            env.place_agent_random_pos()

    return image_list_train, pos_list_train, dir_list_train, image_list_test, pos_list_test, dir_list_test