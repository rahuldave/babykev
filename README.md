# babykev

babykev is a small decision model. You send it one document, the *context*, and a set of typed questions about it. It returns a probability for every possible answer to every question, in one forward pass. It generates no text.

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

A question has one of these types:

| Type | Asks for | The answer has |
|---|---|---|
| `noul` | Yes or no | The probability of yes |
| `choice` | One of several named options | The chosen option, a probability for each option, and a confidence |
| `score` | A level on an ordered scale | The expected level, a probability for each level, and a confidence |

## Where it comes from

babykev comes from the first commit of [kev](https://github.com/jaredpalmer/kev/tree/d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16)
by Jared Palmer. That commit is about 830 lines of Python around one idea. A small language model,
Qwen2.5-0.5B with a LoRA adapter, reads the state and the questions as one sequence. An attention mask
keeps each question from seeing the others, and a pointer head scores every option.

babykev is a repository for teaching. It keeps the model of kev. It builds the engineering around the
model again, from the packaging and the tests to the training and to inference on the command line. The
aim is to show a student the infrastructure and the software development of training a model.

## What you need

Install two tools on your machine, uv and just. The project installs everything else itself.

| Tool | What it is for |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Python itself, the dependencies, and running the program |
| [just](https://just.systems/) | The tasks of the project. `just` lists them |

## Start

Run `just setup` first. It installs Python and every dependency into `.venv`, exactly as `uv.lock` says.
Then run `just test`. `just` lists the tasks, `just help VERB` explains one, and `uv run babykev help` is the
help of the program itself. A recipe gives its extra words to the tool behind it. For example,
`just test -k help` runs only the tests with `help` in their name.

The model needs four packages, torch, transformers, peft and safetensors. torch is the build for the CPU,
from the index of PyTorch that `pyproject.toml` names. On a Mac, the same build runs on the GPU of Apple
too. With them, `.venv` takes about 750 MB, and torch is most of it.

`just smoke` runs the model on this machine, on the best device that it finds. The first time, it
downloads the base model from the Hugging Face Hub, about 1 GB. It answers the five sample requests,
untrained, and prints each answer beside its label.

Run `just hook` once in each clone. It installs a git hook, which runs `just check` before each commit.
`just check` runs every check of the project, and if one fails, git makes no commit. One check is a floor.
The tests must run at least 90% of the code. You can run `just check` yourself at any time.

## What is here

| Path | What it is |
|---|---|
| `pyproject.toml`, `uv.lock` | The project: its name, its Python, its dependencies, the settings of its tools, and the exact versions to install |
| `src/babykev/api.py` | The contract: the request and its three types of question as pydantic models, the text and options the model reads, and the answers made from its probabilities |
| `src/babykev/cli.py` | The command `babykev` |
| `src/babykev/model.py` | The model, a language model with a head that points at the options |
| `src/babykev/smoke.py` | The smoke run, which asks the real model the sample requests on this machine |
| `data/cheese/sample.jsonl` | Five requests about cheese, one on each line. Every question has a `label`, the right answer |
| `tests/unit/` | The tests of the contract, of the command, of the model and of the smoke. One test checks every line of the sample data against the contract |
| `tests/conftest.py` | The fixtures that every test can ask for: the tiny tokenizer, a tiny model and a record |
| `tests/tiny.py` | A small tokenizer and backbone, built in memory, in place of the real ones |
| `docs/` | The docs site. `just docs` builds it into `docs/_site`, with the reference made from the code |
| `justfile` | Every task of the project, one recipe each. `just help` explains them |
| `README.md` | This file |
| `LICENSE`, `NOTICE` | The licence, and the credit to kev |
| `.python-version` | The version of Python that the project uses |
| `.gitignore` | The files that git does not track, for example `.venv` |
| `.pre-commit-config.yaml` | The hooks that git runs before each commit, once `just hook` installs them. The last runs `just check` |
| `.yamllint` | The settings of yamllint, which `just lint-config` runs on every YAML file |

## How the history is organised

The project is built in steps, on the branch `tutorial`. Each step has a tag, `tut-00`, `tut-01` and so
on, and the message of the tag says what the step adds. The commits between two tags are the moves of a
step. The subject of a move starts with its name, for example `tut-01.1:`. The last move of a step is the
commit of its tag.

```
git tag -n1 --list 'tut-*'                   # the steps, one line each
git log --oneline --reverse tut-01           # every move, up to the step tut-01
git log --oneline --reverse tut-00..tut-01   # the moves of one step
git switch --detach tut-00                   # the project as it was at a step
```

Run `git switch` only in a clone of your own. In timewalk, the buttons of the page move the project from step to
step.

## Licence

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The first commit of kev had no licence file.
kev became Apache-2.0 three commits later.
