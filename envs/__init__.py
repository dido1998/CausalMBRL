from gym.envs.registration import register


register(
    'ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

# Weighted Shapes

register(
    'WeightedShapesTrain-v1',
    entry_point='envs.weighted_block_pushing:BlockPushing',
    max_episode_steps=200,
    kwargs={'render_type': 'shapes'},
)

register(
    'WeightedShapesEval-v1',
    entry_point='envs.weighted_block_pushing:BlockPushing',
    max_episode_steps=50,
    kwargs={'render_type': 'shapes'},
)

# Weighted Shapes Environments

def register_shapes(name='WShapes-{}-{}-{}-{}-{}', typ='Observed',
    num_objects=5, mode='Train', cmap='Blues', **kwargs):
    
    register(
        name.format(typ, mode, num_objects, cmap, 'v0'),
        entry_point='envs.weighted_block_pushing:BlockPushing',
        max_episode_steps=200,
        kwargs=dict(
            render_type='shapes',
            num_objects=num_objects,
            mode=mode,
            cmap=cmap,
            typ=typ,
            **kwargs),
    )

def register_shapes_rl(name='WShapesRL-{}-{}-{}-{}-{}', typ='Observed',
    num_objects=5, mode='Train', cmap='Blues', **kwargs):
    
    register(
        name.format(typ, mode, num_objects, cmap, 'v0'),
        entry_point='envs.weighted_block_pushing_rl:BlockPushingRL',
        max_episode_steps=200,
        kwargs=dict(
            render_type='shapes',
            num_objects=num_objects,
            mode=mode,
            cmap=cmap,
            typ=typ,
            **kwargs),
    )

for n_obj in [3,4,5]:
    for mode in ["Train", "Test-v1", "Test-v2", "Test-v3", "ZeroShot", "ZeroShotShape"]:
        for cmap in ["Blues", "Reds", "Greens"]:
            register_shapes('WShapes-{}-{}-{}-{}-{}', 'Observed',
                n_obj, mode, cmap)
            register_shapes_rl('WShapesRL-{}-{}-{}-{}-{}', 'Observed',
                n_obj, mode, cmap)

for n_obj in [3,4,5]:
    for mode in ["Train", "FewShot-v1", "FewShot-v2", "FewShot-v3", "ZeroShotShape"]:
        for cmap in ["Sets", "Pastels"]:
            register_shapes('WShapes-{}-{}-{}-{}-{}', 'Unobserved',
                n_obj, mode, cmap)
            register_shapes_rl('WShapesRL-{}-{}-{}-{}-{}', 'Unobserved',
                n_obj, mode, cmap)

for n_obj in [3,4,5]:
    for mode in ["Train", "FewShot-v1", "FewShot-v2", "FewShot-v3"]:
        for cmap in ["Sets", "Pastels"]:
            register_shapes('WShapes-{}-{}-{}-{}-{}', 'FixedUnobserved',
                n_obj, mode, cmap)
            register_shapes_rl('WShapesRL-{}-{}-{}-{}-{}', 'FixedUnobserved',
                n_obj, mode, cmap)

# Color Changing 
def register_chemistry_envs(name = 'ColorChanging-{}-{}-{}', num_objects = 5,
                            num_colors = 5, **kwargs):
    register(
        name.format(num_objects, num_colors, 'v0'),
        entry_point = 'envs.chemistry_env:ColorChanging',
        max_episode_steps = 200,
        kwargs=dict(
            render_type='shapes',
            num_objects=num_objects,
            num_colors=num_colors,
            **kwargs),
    )

def register_chemistry_envs_moving(name = 'ColorChangingMoving-{}-{}-{}', num_objects = 5,
                            num_colors = 5, **kwargs):
    register(
        name.format(num_objects, num_colors, 'v0'),
        entry_point = 'envs.chemistry_env_moving:ColorChangingMoving',
        max_episode_steps = 200,
        kwargs=dict(
            render_type='shapes',
            num_objects=num_objects,
            num_colors=num_colors,
            **kwargs),
    )

def register_chemistry_rl_envs(name = 'ColorChangingRL-{}-{}-{}-{}-{}', num_objects = 5,
                            num_colors = 5, movement = 'Dynamic', max_steps = 50, **kwargs):
    register(
        name.format(num_objects, num_colors, movement, max_steps, 'v0'),
        entry_point = 'envs.chemistry_env_rl:ColorChangingRL',
        kwargs=dict(
            render_type='shapes',
            num_objects=num_objects,
            num_colors=num_colors,
            movement=movement,
            max_steps=max_steps,
            **kwargs),
    )

for n_obj in [3, 4, 5, 6, 7, 8]:
    for n_colors in [3, 4, 5, 6, 7, 8]:
        register_chemistry_envs('ColorChanging-{}-{}-{}', n_obj, n_colors)

for n_obj in [3, 4, 5, 6, 7, 8]:
    for n_colors in [3, 4, 5, 6, 7, 8]:
        register_chemistry_envs_moving('ColorChangingMoving-{}-{}-{}', n_obj, n_colors)

for n_obj in [3, 4, 5, 6, 7, 8]:
    for n_colors in [3, 4, 5, 6, 7, 8]:
        for movement in ['Dynamic', 'Static']:
            for steps in [10, 20, 30, 40, 50, 200]:
                register_chemistry_rl_envs('ColorChangingRL-{}-{}-{}-{}-{}', n_obj, n_colors, movement, steps)



register(
    'ColorChanging-v0',
    entry_point =  'envs.chemistry_env:ColorChanging',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

# Cubes

register(
    'CubesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)
