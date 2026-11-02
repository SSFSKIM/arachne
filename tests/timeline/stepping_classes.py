"""A dt updater, in a module of its own, that reads a value an instance
inherits from its class (``stepping_scenes.ClassShadow``; the updater that
rebinds it is ``stepping_producer.rebind_source``, in another module, so
that no name of this one's is that one's too).
"""


def read_source(holder):
    """A dt updater integrating ``holder.source``, which ``holder``'s class
    or one of its bases holds.
    """
    return lambda m, dt: m.increment_value(dt * holder.source.get_value())
