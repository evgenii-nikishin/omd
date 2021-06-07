import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'omd'

    config.model_lr = 3e-4
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.model_hidden_dim = 64  # for the misspecification experiments

    config.discount = 0.99

    config.outer_steps = 1
    config.inner_steps = 1

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None

    config.replay_buffer_size = None

    return config
