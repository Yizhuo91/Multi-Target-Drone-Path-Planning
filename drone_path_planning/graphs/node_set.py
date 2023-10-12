from typing import Any
from typing import NamedTuple

from drone_path_planning.graphs.component_set import ComponentSet


class _NodeSetTuple(NamedTuple):
    features: Any


class NodeSet(ComponentSet, _NodeSetTuple):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

    @property
    def _features(self):
        return self.features
