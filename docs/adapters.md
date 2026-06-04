# Writing custom adapters

An adapter is the piece of diapason that turns an evaluation example into a
model prediction. Two adapters ship with the package — the deterministic
`dummy` adapter and the `http` adapter for OpenAI-compatible endpoints — but
most evaluations need their own: a local model, a vendor SDK, or a script that
replays stored outputs.

## The adapter contract

Subclass `ModelAdapter`, set a unique `name`, and implement `predict`:

```python
from diapason.adapters.base import ModelAdapter
from diapason.datasets.manifest import Example


class EchoUpperAdapter(ModelAdapter):
    """Predict the reference text in upper case."""

    name = "echo_upper"

    def predict(self, example: Example) -> str:
        return (example.reference_text or "").upper()
```

`predict` receives an `Example` — one row of the manifest — and returns the
model's text answer. Whatever it raises is recorded as an error for that
example, so a flaky example does not abort the whole run.

The fields available on `Example` depend on the task type:

| Field | Meaning |
| --- | --- |
| `example.id` | The manifest row id. |
| `example.audio` | Audio file path, when the manifest has one. |
| `example.prompt` | The question or instruction (AQA calls it `question`). |
| `example.reference_text` | The reference transcription or answer. |
| `example.meta` | Any extra manifest fields, keyed by name. |

## Registering the adapter

Register the class under the name suite configs will use:

```python
from diapason.registry import adapters

adapters.register("echo_upper")(EchoUpperAdapter)
```

The decorator form is equivalent and more common:

```python
@adapters.register("echo_upper")
class EchoUpperAdapter(ModelAdapter): ...
```

Registration happens at import time, so make sure the module defining the
adapter is imported before the suite runs. When you drive diapason from your
own script this is automatic; when you use the CLI, put the adapter in a
small wrapper module:

```python
# run_with_echo.py
import my_adapters  # registers echo_upper
from diapason.cli import main

if __name__ == "__main__":
    main()
```

## Constructor options

The runner constructs your adapter with the `adapter_options` mapping from the
suite config, so constructor keyword arguments are part of your public
interface:

```yaml
tasks:
  - name: transcription
    type: asr
    manifest: data/asr.jsonl
    adapter: echo_upper
    adapter_options:
      suffix: "!"
```

Keep defaults for every option so the adapter works with an empty
`adapter_options` mapping.

## Testing an adapter

Adapters are easiest to test without the runner:

```python
def test_echo_upper() -> None:
    example = Example(id="1", reference_text="the cat sat")
    assert EchoUpperAdapter().predict(example) == "THE CAT SAT"
```

For adapters that call a network endpoint, test the request/response helpers
in isolation (as the `http` adapter's tests do) rather than hitting a live
service.
