import os
from typing import Optional
from typing import Tuple

import tensorflow as tf

from drone_path_planning.agents import MultipleChasersSingleTargetAgent
from drone_path_planning.environments import MultipleChasersSingleMovingTargetEnvironment
from drone_path_planning.evaluators import Evaluator
from drone_path_planning.evaluators import MultiAgentDeepQNetworkEvaluator
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.scenarios.scenario import Scenario
from drone_path_planning.trainers import MultiAgentDeepQNetworkTrainer
from drone_path_planning.trainers import Trainer
from drone_path_planning.utilities.agent_groups import MultiAgentGroup
from drone_path_planning.utilities.agent_groups import MultiAgentTrainingGroup
from drone_path_planning.utilities.constants import AGENT
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import REST
from drone_path_planning.utilities.training_helpers import ReplayBuffer


_CHASERS = 'chasers'


_CHASER_0 = 'chaser_0'
_CHASER_1 = 'chaser_1'
_CHASER_2 = 'chaser_2'
_CHASER_3 = 'chaser_2'


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


class FourChasersSingleMovingTargetScenario(Scenario):
    def create_trainer(self, save_dir: str, logs_dir: Optional[str] = None) -> Trainer:
        chaser_agent: MultipleChasersSingleTargetAgent
        try:
            chaser_agent_save_dir = os.path.join(save_dir, _CHASERS)
            chaser_agent = tf.keras.models.load_model(chaser_agent_save_dir)
        except IOError:
            chaser_agent = MultipleChasersSingleTargetAgent(
                output_specs=OutputGraphSpec(
                    node_sets={
                        AGENT: [
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
        chaser_agent_ids = {
            _CHASER_0,
            _CHASER_1,
            _CHASER_2,
            _CHASER_3,
        }
        environment = MultipleChasersSingleMovingTargetEnvironment(chaser_ids=chaser_agent_ids)
        replay_buffer = ReplayBuffer()
        validation_environment = MultipleChasersSingleMovingTargetEnvironment(chaser_ids=chaser_agent_ids)
        groups = {
            _CHASERS: MultiAgentTrainingGroup(
                agent=chaser_agent,
                agent_compile_kwargs=dict(
                    optimizer=optimizer,
                ),
                agent_ids=chaser_agent_ids,
                replay_buffer=replay_buffer,
                replay_buffer_size=_REPLAY_BUFFER_SIZE,
            ),
        }
        trainer = MultiAgentDeepQNetworkTrainer(
            groups=groups,
            environment=environment,
            num_epochs=_NUM_EPOCHS,
            num_steps_per_epoch=_NUM_STEPS_PER_EPOCH,
            max_num_steps_per_episode=_MAX_NUM_STEPS_PER_EPISODE,
            save_dir=save_dir,
            validation_environment=validation_environment,
            num_val_episodes=_NUM_VAL_EPISODES,
            max_num_steps_per_val_episode=_MAX_NUM_STEPS_PER_VAL_EPISODE,
            logs_dir=logs_dir,
        )
        return trainer

    def create_evaluator(self, save_dir: str, plot_data_dir: str, logs_dir: Optional[str] = None) -> Evaluator:
        chaser_agent: MultipleChasersSingleTargetAgent
        try:
            chaser_agent_save_dir = os.path.join(save_dir, _CHASERS)
            chaser_agent = tf.keras.models.load_model(chaser_agent_save_dir)
        except IOError:
            chaser_agent = MultipleChasersSingleTargetAgent(
                output_specs=OutputGraphSpec(
                    node_sets={
                        AGENT: [
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
        chaser_agent_ids = {
            _CHASER_0,
            _CHASER_1,
            _CHASER_2,
            _CHASER_3,
        }
        environment = MultipleChasersSingleMovingTargetEnvironment(chaser_ids=chaser_agent_ids)
        groups = {
            _CHASERS: MultiAgentGroup(
                agent=chaser_agent,
                agent_compile_kwargs=dict(),
                agent_ids=chaser_agent_ids,
            ),
        }
        evaluator = MultiAgentDeepQNetworkEvaluator(
            groups=groups,
            environment=environment,
            plot_data_dir=plot_data_dir,
            num_episodes=_NUM_EVAL_EPISODES,
            max_num_steps_per_episode=_MAX_NUM_STEPS_PER_EVAL_EPISODE,
            logs_dir=logs_dir,
        )
        return evaluator
