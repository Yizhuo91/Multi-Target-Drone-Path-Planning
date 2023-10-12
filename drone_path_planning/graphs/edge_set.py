from typing import Any
from typing import NamedTuple

import tensorflow as tf

from drone_path_planning.graphs.component_set import ComponentSet


class _EdgeSetTuple(NamedTuple):
    features: Any
    senders: tf.Tensor
    receivers: tf.Tensor
    sender_set: str
    receiver_set: str


class EdgeSet(ComponentSet, _EdgeSetTuple):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

    @property
    def _features(self):
        return self.features
