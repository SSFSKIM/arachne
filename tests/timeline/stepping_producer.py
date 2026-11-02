"""A dt-free updater that writes a module attribute it imports under an
alias of its own (``stepping_scenes.ModuleAlias``).
"""

from tests.timeline import stepping_shared as storage


def publish(t):
    """An updater that copies the tracker's value into the shared module."""

    def update(m):
        storage.value = t.get_value()

    return update
