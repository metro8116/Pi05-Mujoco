"""Expose simulation utilities without eagerly initializing OpenGL.

Provide package-level access to the MuJoCo parser through a lazy import.
This lets configuration and command-line code import the package before
selecting a rendering backend. The parser is loaded when requested;
importing this package alone does not create a scene or viewer.
"""
__all__ = ["MuJoCoParserClass"]


def __getattr__(name):
    if name == "MuJoCoParserClass":
        from .mujoco_parser import MuJoCoParserClass
        globals()[name] = MuJoCoParserClass
        return MuJoCoParserClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
