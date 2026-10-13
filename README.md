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
Then `just` lists the tasks, `just help VERB` explains one, and `uv run babykev help` is the help of the
program itself.

## What is here

| Path | What it is |
|---|---|
| `pyproject.toml`, `uv.lock` | The project: its name, its Python, its dependencies, and the exact versions to install |
| `src/babykev/` | The code. `cli.py` is the command `babykev` |
| `justfile` | Every task of the project, one recipe each. `just help` explains them |
| `README.md` | This file |
| `LICENSE`, `NOTICE` | The licence, and the credit to kev |
| `.python-version` | The version of Python that the project uses |
| `.gitignore` | The files that git does not track, for example `.venv` |

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
