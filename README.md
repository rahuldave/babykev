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

Three tools on the machine. This project installs everything else itself.

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
`docs/_site/index.html`. Under Guide, two pages explain the model: the four ideas, and the mask as a
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

`just types` is ty, the type checker: it reads every call against what the called function declares,
and it is one of the lines of `just check`, so the hook runs it on every commit.

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

## Licence

Apache-2.0: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The original project kev's first commit carried no licence file; kev became Apache-2.0 three commits later.
