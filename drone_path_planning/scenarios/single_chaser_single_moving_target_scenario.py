from typing import Optional
from typing import Tuple

import tensorflow as tf

from drone_path_planning.agents import SingleChaserSingleTargetAgent
from drone_path_planning.environments import SingleChaserSingleMovingTargetEnvironment
from drone_path_planning.evaluators import Evaluator
from drone_path_planning.evaluators import SingleAgentDeepQNetworkEvaluator
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.scenarios.scenario import Scenario
from drone_path_planning.trainers import SingleAgentDeepQNetworkTrainer
from drone_path_planning.trainers import Trainer
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import REST
from drone_path_planning.utilities.training_helpers import ReplayBuffer


_INITIAL_LEARNING_RATE: float = 1e-5
_NUM_STEPS_PER_FULL_LEARNING_RATE_DECAY: int = 524288
_LEARNING_RATE_DECAY_RATE: float = 0.9999956082
_NUM_ITERATIONS: int = 2097152
_MAX_NUM_STEPS_PER_EPISODE: int = 256
_NUM_VAL_EPISODES: int = 16
_MAX_NUM_STEPS_PER_VAL_EPISODE: int = _MAX_NUM_STEPS_PER_EPISODE
_NUM_STEPS_PER_EPOCH: int = _MAX_NUM_STEPS_PER_EPISODE * 8
_NUM_EPOCHS: int = _NUM_ITERATIONS // _NUM_STEPS_PER_EPOCH
_REPLAY_BUFFER_SIZE: int = _MAX_NUM_STEPS_PER_EPISODE * 64
_NUM_EVAL_EPISODES: int = 16
_MAX_NUM_STEPS_PER_EVAL_EPISODE: int = _MAX_NUM_STEPS_PER_EPISODE


class SingleChaserSingleMovingTargetScenario(Scenario):
    def create_trainer(self, save_dir: str, logs_dir: Optional[str] = None) -> Trainer:
        agent: SingleChaserSingleTargetAgent
        try:
            agent = tf.keras.models.load_model(save_dir)
        except IOError:
            agent = SingleChaserSingleTargetAgent(
                output_specs=OutputGraphSpec(
                    node_sets={
                        'self': [
                            {
                                REST: 1,
                                FORWARD: 1,
                                BACKWARD: 1,
                                ANTI_CLOCKWISE: 1,
                                CLOCKWISE: 1,
                            },
                        ],
                    },
                    edge_sets=dict(),
                ),
                latent_size=128,
                num_hidden_layers=2,
                num_message_passing_steps=1,
                tau=0.08,
            )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=_INITIAL_LEARNING_RATE,
                decay_steps=_NUM_STEPS_PER_FULL_LEARNING_RATE_DECAY,
                decay_rate=_LEARNING_RATE_DECAY_RATE,
            ),
        )
        environment = SingleChaserSingleMovingTargetEnvironment()
        replay_buffer = ReplayBuffer()
        validation_environment = SingleChaserSingleMovingTargetEnvironment()
        trainer = SingleAgentDeepQNetworkTrainer(
            agent=agent,
            optimizer=optimizer,
            environment=environment,
            replay_buffer=replay_buffer,
            replay_buffer_size=_REPLAY_BUFFER_SIZE,
            num_epochs=_NUM_EPOCHS,
            num_steps_per_epoch=_NUM_STEPS_PER_EPOCH,
            max_num_steps_per_episode=_MAX_NUM_STEPS_PER_EPISODE,
            save_dir=save_dir,
            logs_dir=logs_dir,
            validation_environment=validation_environment,
            num_val_episodes=_NUM_VAL_EPISODES,
            max_num_steps_per_val_episode=_MAX_NUM_STEPS_PER_VAL_EPISODE,
        )
        return trainer

    def create_evaluator(self, save_dir: str, plot_data_dir: str, logs_dir: Optional[str] = None) -> Evaluator:
        agent: SingleChaserSingleTargetAgent
        try:
            agent = tf.keras.models.load_model(save_dir)
        except IOError:
            agent = SingleChaserSingleTargetAgent(
                output_specs=OutputGraphSpec(
                    node_sets={
                        'self': [
                            {
                                REST: 1,
                                FORWARD: 1,
                                BACKWARD: 1,
                                ANTI_CLOCKWISE: 1,
                                CLOCKWISE: 1,
                            },
                        ],
                    },
                    edge_sets=dict(),
                ),
                latent_size=128,
                num_hidden_layers=2,
                num_message_passing_steps=1,
                tau=0.08,
            )
        environment = SingleChaserSingleMovingTargetEnvironment()
        evaluator = SingleAgentDeepQNetworkEvaluator(
            agent=agent,
            environment=environment,
            plot_data_dir=plot_data_dir,
            num_episodes=_NUM_EVAL_EPISODES,
            max_num_steps_per_episode=_MAX_NUM_STEPS_PER_EVAL_EPISODE,
            logs_dir=logs_dir,
        )
        return evaluator
