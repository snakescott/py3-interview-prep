# A python environment for programming interviews
**Heavily WIP until this disclaimer is removed**

You can use this boilerplate however you want but it was designed to support roughly the single file workflow suggested by [@asottile](https://www.github.com/asottile) in [how to ace the coding interview (intermediate) anthony explains #358](https://www.youtube.com/watch?v=eVNkO6g0fP8). 

`interview/code.py` has been seeded with a few Internet-sourced problems and solutions to show the boilerplate by example and hopefully allow you to focus productively on "step six: implement the thing" (see the video!).

## Quickstart
### Configuration
1. Clone the repo somewhere
1. Ensure `uv` is installed correctly by confirming `uv --help` prints expected output.
1. Run `uv sync` to configure toolchain and dependencies
1. Confirm `uv run pytest` runs tests in `interview/code.py` and that they pass.
1. (Optional) open the checkout in Visual Studio Code and run tests from inside the IDE.

### Devloop in interview
1. Develop inside `interview/code.py`
2. Use pytest to ensure tests pass, e.g. `uv run pytest`

## Credits and other notes
* [@asottile](https://www.github.com/asottile) for the video/content/expertise that inspired this repo
* [@bcorfman](https://www.github.com/bcorfman) for [bcorfman/ultraviolet](https://github.com/bcorfman/ultraviolet) which I used to boostrap the repo

A note on the use of `uv`: `uv` builds on top of a bunch of open source work on Python tooling and [that community has mixed feelings](https://www.youtube.com/watch?v=XzW4-KEB664) about how derivatives were built on top of that work (to be clear, as allowed license) and commercialized into @astral-sh, which went on to raise millions of dollars in venture capital. I believe `uv` makes this repo easier to use than alternatives, but hope that Astral finds a way to give back to the Python toolchain development community in the future.


## TODO
* Add more examples to `code.py` (and update credits/LICENSE as appropriate)
* Add and document more batteries, like `more_itertools`
* Integrate with some sort of CI/CD (likely Github Actions) to ensure that tests in `code.py` stay green

