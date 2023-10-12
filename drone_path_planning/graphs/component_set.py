from __future__ import annotations
import abc


class ComponentSet:
    @abc.abstractmethod
    def replace(self, **kwargs):
        raise NotImplementedError()

    def map(self, features_map_function=None):
        mapped_features = self._features if features_map_function is None else features_map_function(self._features)
        mapped_component_set = self.replace(features=mapped_features)
        return mapped_component_set

    @property
    @abc.abstractmethod
    def _features(self):
        raise NotImplementedError()

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError('unsupported operand type(s) for +: \'{self_type}\' and \'{other_type}\''.format(
                self_type=type(self).__name__,
                other_type=type(other).__name__,
            ))
        new_features = self._features + other._features
        new_component_set = self.replace(features=new_features)
        return new_component_set
