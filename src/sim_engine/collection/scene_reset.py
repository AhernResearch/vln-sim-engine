"""Testable reset boundary; scene ownership and fixed physics belong to its worker."""


def reset_or_load(env, episode, camera, seed, state_factory, quaternion, position):
    if env.sim is None:
        env.reset(episode, camera, seed)
        return True
    # Scene and camera/physics configuration are fixed for the lifetime of the backend.
    env.sim.reset()
    env.sim.seed(seed)
    state = state_factory()
    state.position = position(episode["start_position"])
    state.rotation = quaternion(episode["start_rotation"])
    env.sim.initialize_agent(0, state)
    return False
