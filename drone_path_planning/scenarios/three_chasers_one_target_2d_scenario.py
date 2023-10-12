import os
from typing import Optional

import tensorflow as tf

from drone_path_planning.agents import MultipleChasersSingleTargetAgent
from drone_path_planning.environments import VariableNumberChasersSingleMovingTargetEnvironment
from drone_path_planning.evaluators import Evaluator
from drone_path_planning.evaluators import VariableNumberAgentDeepQNetworkEvaluator
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.scenarios.scenario import Scenario
from drone_path_planning.trainers import Trainer
from drone_path_planning.trainers import VariableNumberAgentDeepQNetworkTrainer
from drone_path_planning.utilities.agent_groups import VariableNumberAgentGroup
from drone_path_planning.utilities.agent_groups import VariableNumberAgentTrainingGroup
from drone_path_planning.utilities.constants import AGENT
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import REST
from drone_path_planning.utilities.functions import create_time_step_spec
from drone_path_planning.utilities.functions import create_transition_spec
from drone_path_planning.utilities.training_helpers import ReplayBuffer


_CHASERS = 'chasers'


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


class ThreeChasersOneTarget2DScenario(Scenario):
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
        group_min_max_nums = {
            _CHASERS: (3, 3),
        }
        environment = VariableNumberChasersSingleMovingTargetEnvironment(group_min_max_nums=group_min_max_nums)
        replay_buffer = ReplayBuffer()
        validation_environment = VariableNumberChasersSingleMovingTargetEnvironment(group_min_max_nums=group_min_max_nums)
        groups = {
            _CHASERS: VariableNumberAgentTrainingGroup(
                agent=chaser_agent,
                agent_compile_kwargs=dict(
                    optimizer=optimizer,
                ),
                replay_buffer=replay_buffer,
                replay_buffer_size=_REPLAY_BUFFER_SIZE,
            ),
        }
        time_step_spec = create_time_step_spec(environment.get_observation_spec())
        tf_collect_step_input_signatures = {
            _CHASERS: (time_step_spec,),
        }
        transition_spec = create_transition_spec(environment.get_observation_spec())
        tf_train_step_input_signatures = {
            _CHASERS: (transition_spec,),
        }
        val_time_step_spec = create_time_step_spec(validation_environment.get_observation_spec())
        tf_predict_step_input_signatures = {
            _CHASERS: (val_time_step_spec,),
        }
        trainer = VariableNumberAgentDeepQNetworkTrainer(
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
            tf_collect_step_input_signatures=tf_collect_step_input_signatures,
            tf_train_step_input_signatures=tf_train_step_input_signatures,
            tf_predict_step_input_signatures=tf_predict_step_input_signatures,
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
        group_min_max_nums = {
            _CHASERS: (3, 3),
        }
        environment = VariableNumberChasersSingleMovingTargetEnvironment(group_min_max_nums=group_min_max_nums)
        groups = {
            _CHASERS: VariableNumberAgentGroup(
                agent=chaser_agent,
                agent_compile_kwargs=dict(),
            ),
        }
        time_step_spec = create_time_step_spec(environment.get_observation_spec())
        tf_predict_step_input_signatures = {
            _CHASERS: (time_step_spec,),
        }
        evaluator = VariableNumberAgentDeepQNetworkEvaluator(
            groups=groups,
            environment=environment,
            plot_data_dir=plot_data_dir,
            num_episodes=_NUM_EVAL_EPISODES,
            max_num_steps_per_episode=_MAX_NUM_STEPS_PER_EVAL_EPISODE,
            logs_dir=logs_dir,
            tf_predict_step_input_signatures=tf_predict_step_input_signatures,
        )
        return evaluator
