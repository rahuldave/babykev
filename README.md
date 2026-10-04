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
| [Quarto](https://quarto.org/) | The docs site |

Then `just setup`. `just` lists the tasks, `just help VERB` explains one, and `uv run babykev help` is
the program's own help.

## What is here

| Path | What it is |
|---|---|
| `pyproject.toml`, `uv.lock` | The project: its name, its Python, its dependencies, and the exact versions installed |
| `src/babykev/` | The code. `cli.py` is the `babykev` command |
| `justfile` | Every task of the project, one recipe each. `just help` explains them |
| `LICENSE`, `NOTICE` | The licence, and the credit to kev |
| `.python-version` | The Python version the project uses |
| `.gitignore` | Everything a training run will write, ignored from the first commit on |

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
