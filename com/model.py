from abc import abstractmethod, ABC


from layer import Layer


class Model:
    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def __init__(self) -> None:
        self._layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
