# Arachne

Arachne is a visual engine for explanatory math videos. It is a fork of
[Manim Community Edition](https://github.com/ManimCommunity/manim) at v0.22.1, and it is being rebuilt in
stages. Until a stage replaces it, a part of the engine works as it does in Manim Community Edition, and that
project's documentation describes it.

The package imports as `arachne` and its command is `arachne`:

```sh
pip install git+https://github.com/SSFSKIM/arachne@main
arachne -ql example_scenes/basic.py SquareToCircle
```

```python
from arachne import *


class SquareToCircle(Scene):
    def construct(self):
        self.play(Create(Square()))
```

## License

No license is granted for Arachne's own work. The code inherited from Manim Community Edition and 3Blue1Brown's
Manim keeps its MIT licenses under `LICENSES/`. See [NOTICE.md](NOTICE.md).
