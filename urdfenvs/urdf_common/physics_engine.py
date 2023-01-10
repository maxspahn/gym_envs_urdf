class PhysicsEngine(object):
    def __init__(self, render: bool):
        self._render = render


    def render(self):
        return self._render
