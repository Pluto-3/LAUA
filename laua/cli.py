"""Entry point — starts the Textual UI."""

from __future__ import annotations


def main() -> None:
    from laua.ui.app import LauaApp
    app = LauaApp()
    app.run()


if __name__ == "__main__":
    main()
