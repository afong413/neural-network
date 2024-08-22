#!/usr/bin/env python3


def render(image, shape, label=None, correct=None):
    """
    Render an image to the console. I decided against including this in
    `model.py`, because it slowed down testing too much.
    """
    if label is not None:
        print(label)
    for i in range(shape[0]):
        for j in range(shape[1]):
            z = int(image[shape[1] * i + j])

            print(  # Weird ANSI stuff that I have to look up in a table
                f"\u001b[48;2;{0 if correct else z};"
                f"{0 if correct is False else z};"
                f"{z if correct is None else 0}m  \u001b[0m",
                end="",
            )

        print()
