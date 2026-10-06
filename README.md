# babykev

babykev is a small decision model. You send it one document, the *context* (the `state` field of a request), and a set of typed questions about it. It returns a probability for every possible answer to every question, in one forward pass. It generates no text.

A request looks like this:

```json
{
  "state": "Ardsallagh smoked cheese. Carrigtwohill. Natural rind. Smooth. White. Taste: mild, nutty, smokey and sweet.",
  "questions": {
    "has_rind": {"type": "noul", "instructions": "Is there a rind on this cheese?"},
    "milk": {
      "type": "choice",
      "instructions": "What kind of milk goes into this cheese?",
      "criteria": {"cow": null, "goat": null, "sheep": null, "buffalo": null, "mixed": "More than one kind of milk"}
    },
    "firmness": {
      "type": "score",
      "instructions": "Where does this cheese sit between soft and hard?",
      "criteria": ["Soft", "Semi-soft", "Semi-hard", "Hard"]
    }
  }
}
```

There are three types of question:

| Type | Asks for | The answer holds |
|---|---|---|
| `noul` | Yes or no | The probability of yes |
| `choice` | One of several named options | The chosen option, a probability for each option, and a confidence |
| `score` | A level on an ordered scale | The expected level, a probability for each level, and a confidence |

## Where it comes from

babykev is derived from the first commit of [kev](https://github.com/jaredpalmer/kev/tree/d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16)
by Jared Palmer. That commit is about 830 lines of Python around one idea: a small language model
(Qwen2.5-0.5B with a LoRA adapter) reads the state and the questions as one sequence, an attention mask
keeps each question from seeing the others, and a pointer head scores every option.

This is an educational repo. The repo `babykev` keeps the original model but rebuilds the engineering around it: packaging, tests, types, docs, training, experiment tracking, and simple inference on the command line. The idea is to expose to a student the infrastructural and software development aspects of, for now, training a model.

## What you need

Three tools on the machine. This project installs everything else itself. On a Mac or on Linux,
install them directly. On Windows, work inside WSL2, which is Linux, or use the dev image (below);
the recipes are not written for Windows itself.

| Tool | What it is for |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Python itself, the dependencies, and running the program |
| [just](https://just.systems/) | The project's tasks. `just` lists them |
| [Quarto](https://quarto.org/) | The docs site: `just docs`. `just setup` says if it is missing |

Then `just setup`, and `just test`. `just setup` installs torch's CPU build from PyTorch's own index,
which is enough for the tests and for running the model on a laptop. `just fmt` formats the code and `just lint` checks it against the
rules in `pyproject.toml`. `just check` runs everything a commit must pass, and `just setup` installs it
as a git hook, so a commit that fails a check is refused. `just` lists the tasks, `just help VERB` explains one, and `uv run
babykev help` is the program's own help. Extra arguments go to the tool behind a recipe: `just test -k
render` runs only the tests whose name contains `render`.

The documentation comes from the code. `just docs` builds the site into `docs/_site/`; open
`docs/_site/index.html`. `just docs publish` builds it and pushes it to the `gh-pages` branch of the
GitHub repository that `origin` names, for GitHub Pages to serve (once, in the repository's settings,
Pages must be set to deploy from that branch); with no remote, or one that is not on GitHub, it says so
and does nothing. Under Guide, two pages explain the model: the four ideas, and the mask as a
grid (a notebook, committed with its outputs; a test re-runs its cells and checks the outputs are still
true). The same text is there at the terminal as plain Markdown, for people and for
agents: `just docs list` names every module, class and function with its summary, `just docs module api`
prints one module, and `just docs check` counts what is typed and documented and lists every gap. The
convention: every function is typed, has a docstring, and carries a docment, a comment beside each
parameter and the return, which the formatter and the linter leave alone as long as the line fits in 100
characters. `src/babykev/docs.py` is the example to read. `just docs check` is part of `just check`, so a
commit that leaves a gap is refused by the hook.

## Running it

`just smoke local` runs everything that exists so far, end to end, on this machine: it loads the base
model, Qwen2.5-0.5B, with a fresh adapter and an untrained pointer head, asks it the questions in the
sample data, and prints each answer beside its label. Nothing is trained yet, so the answers are guesses;
what the run shows is that the pieces fit. The first run downloads the base model from the Hugging Face
Hub, about 1 GB, into its cache; later runs read it from there. The platform is a word of the recipe, and
only `local` exists so far.

`just test` runs every test that needs nothing but our code. One more test, in
`tests/system/`, loads the real model the way the smoke does; it needs the Hub, the weights and a
device, so it runs only when asked for: `just test -m system`.

`just types` is ty, the type checker: it reads every call against what the called function declares,
and it is one of the lines of `just check`, so the hook runs it on every commit.

## In Docker

The `Dockerfile` builds two images from the lockfile, for any machine that has Docker (Docker Desktop
on a Mac or on Windows, OrbStack on a Mac, Docker Engine on Linux or on a cloud VM):

| Image | Holds | Starts | For |
|---|---|---|---|
| dev | Python, every dependency with the dev tools, uv, just, Quarto, git | a shell | working on the project on any system, and checking it the way a CI runner would |
| run | Python and the dependencies the program needs, nothing else | `babykev` | a platform that runs the program |

```
just image build dev        # docker build --target dev, named babykev-dev:COMMIT
just image shell            # a shell in a new container, with this folder mounted at /work: just check
just image start            # or keep one running in the background; then just image shell goes into it,
just image stop             #   and what you installed or ran there is still there, until stop removes it
just image build run
just image run smoke        # the smoke in the run image, on the CPU
```

Each image is built for the machine's own architecture; `just image build run --platform linux/amd64`
builds the one most platforms run. The environment inside lives in `/opt/venv`, so a mounted folder's
own `.venv` is never used. Both images read the Hugging Face cache of the machine, mounted at `/hf`, so
the base model is downloaded once. Extra arguments go to docker: `just image shell --network none`
starts the shell with no network at all, where `just check` still passes and the real model can load
only from the cache. The dev image declares port 8765, the step browser's: `just image shell` and `just image start`
publish it on the first free port here from 8765 up, and say which. The recipe runs on the machine,
never inside an image.

## What is here

| Path | What it is |
|---|---|
| `pyproject.toml`, `uv.lock` | The project: its name, its Python, its dependencies, the settings of its tools (the formatter's, and every lint rule with its reason), and the exact versions installed |
| `src/babykev/api.py` | The contract: the request and its three question types as Pydantic models, how they become the text and options the model sees, and how probabilities become answers |
| `src/babykev/cli.py` | The `babykev` command: `docs` and `smoke` |
| `src/babykev/model.py` | The model: packing a context and its questions into one sequence, the mask that keeps questions apart, the pointer head that scores options, and the class that puts them on a language model |
| `src/babykev/smoke.py` | The smoke run behind `just smoke` |
| `src/babykev/docs.py` | What the code says about itself, as plain Markdown: `babykev docs list`, `docs module NAME`, `docs check` |
| `docs/` | The docs site: `_quarto.yml`, the front page `index.qmd`, the guide pages under `guide/`, `apidocs.py` (writes the reference pages from `babykev docs` before every render) and `linkify.lua` (a name in backticks becomes a link). The built site in `_site/` is not committed |
| `data/cheese/sample.jsonl` | Five lines of data. Each is a request about a cheese with a `label` on every question: the right answer |
| `Dockerfile`, `.dockerignore` | The two images, dev and run, and what a build never copies into them |
| `tests/system/` | The test that is not hermetic: the real model, from the Hub, on this machine's device |
| `tests/unit/` | Unit tests of the contract, of the command, of the documentation, of the model and of the smoke. One test checks every line of the sample data against the contract |
| `tests/tiny.py`, `tests/conftest.py` | A tokenizer with one token per character and a two-layer backbone with random weights, built in memory: the model's tests need no download and no GPU |
| `justfile` | Every task of the project, one recipe each. `just help` explains them |
| `LICENSE`, `NOTICE` | The licence, and the credit to kev |
| `.python-version` | The Python version the project uses |
| `.gitignore` | Everything a training run will write, ignored from the first commit on |
| `.pre-commit-config.yaml`, `.yamllint` | The git hook: what every commit must pass; and the settings of the YAML linter |

## How the history is organised

The project is built in steps. Each step is one commit on `main` with a tag, `step-00`, `step-01` and so
on, and the tag's message says what the step adds.

```
git tag -n1 --list 'step-*'     # the steps, one line each
git show step-00                # a step's full note, and what it changed
git switch --detach step-00     # the project as it was at that step
```

<!-- steps:start -->

The commit of each step is linked here. On GitHub it shows what the step changed. `browse` shows the project as it was at the step.

| Step | Commit | Code at the step | What it adds |
|---|---|---|---|
| `step-00` | [`ee96429`](https://github.com/rahuldave/babykev/commit/ee96429002cd04cb95e86e9a77e9c3d02a0e9497) | [browse](https://github.com/rahuldave/babykev/tree/step-00) | Where we start |
| `step-01` | [`9cc0343`](https://github.com/rahuldave/babykev/commit/9cc0343527b93a817207a11a2866752807f0013d) | [browse](https://github.com/rahuldave/babykev/tree/step-01) | The contract: api.py, and its tests |
| `step-02` | [`4681f25`](https://github.com/rahuldave/babykev/commit/4681f251318456e721cb4a7dcecbb9f73d3589cf) | [browse](https://github.com/rahuldave/babykev/tree/step-02) | Fix the two test errors, and format and lint |
| `step-03` | [`651cd62`](https://github.com/rahuldave/babykev/commit/651cd62cf2d09d53225ed11c9befd7c2316ec1aa) | [browse](https://github.com/rahuldave/babykev/tree/step-03) | Clean code: formatted, linted, typed, docstrings, import order, test coverage |
| `step-04` | [`d27a7f1`](https://github.com/rahuldave/babykev/commit/d27a7f152ade46e9024f0b97481b85fcce07a16c) | [browse](https://github.com/rahuldave/babykev/tree/step-04) | Documentation from the code |
| `step-04b` | [`a9a391d`](https://github.com/rahuldave/babykev/commit/a9a391de1f271392a948cdd16afe2f3b1cfdd437) | [browse](https://github.com/rahuldave/babykev/tree/step-04b) | Every function documented |
| `step-05` | [`46cf862`](https://github.com/rahuldave/babykev/commit/46cf8625a031e10049355df4c37abcb7af58d847) | [browse](https://github.com/rahuldave/babykev/tree/step-05) | A second file: model.py |
| `step-05a` | [`c954af4`](https://github.com/rahuldave/babykev/commit/c954af4beae61eaab89bcb2d4724eea4f2706703) | [browse](https://github.com/rahuldave/babykev/tree/step-05a) | The tests that could tell |
| `step-05b` | [`47b43c0`](https://github.com/rahuldave/babykev/commit/47b43c0916ac186e3efdf42e15dbcf1f04edb1a0) | [browse](https://github.com/rahuldave/babykev/tree/step-05b) | Two images from one lockfile |

<!-- steps:end -->

## Licence

Apache-2.0: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The original project kev's first commit carried no licence file; kev became Apache-2.0 three commits later.
