"""Dt-free updaters that write what another module's code reads: a module
attribute imported under an alias of its own (``stepping_scenes.ModuleAlias``),
and a class attribute an instance inherits (``stepping_scenes.ClassShadow``).
"""

from tests.timeline import stepping_shared as storage


def publish(t):
    """An updater that copies the tracker's value into the shared module."""

    def update(m):
        storage.value = t.get_value()

    return update


def rebind_source(cls, other, scene):
    """An updater that, a quarter of the way in, binds ``cls.source`` to
    ``other``.
    """

    def update(m):
        if scene.time >= 0.25:
            cls.source = other

    return update
