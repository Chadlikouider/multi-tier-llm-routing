# Quality Time

Code for the paper "[Carbon-Aware Quality Adaptation for Energy-Intensive Services](https://arxiv.org/pdf/2411.19058)".

The datasets and full instructions on how to reproduce all experiments can be provided upon request. We will update the repository with the datasets and instructions in the future.


## Setup

```bash
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run experiments using `python main.py`, where you can override individual parameterizations using [hydra](https://hydra.cc/docs/). For example:
```shell
> python main.py optimizer=qt_online region=CISO vp=1,24,168 qor_target=.5 requests_dataset=wiki_en,wiki_de result_dir=final_results
```
Would run 6 experiments and store the results in `./final_results`: three validity periods (`vp`) on two datasets (`requests_dataset`) with the `qt_online` optimizer in the `CISO` region and a minimum quality of responses (`qor_target`) of 0.5.

Full configuration:

```shell
> python main.py --help
main is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

optimizer: greedy, qt, qt_online
user_groups: p0, p10, p100, p20, p30, p40, p5, p50, p60, p70, p80, p90, p95, p96, p97, p98, p99


== Config ==
Override anything in the config (foo.bar=value)

seed: 0
qor_target: 0.5
requests_dataset: wiki_en
region: CISO
vp: 24
model_qualities:
- bad
- good
machines:
- _target_: src.scenario.Machine
  name: AWS_p4d.24xlarge
  power_usage: 3.7818
  pue: 1
  performance:
    llama8B: 41710
    good: 18176
  embedded_carbon: 135
result_dir: results
optimizer:
  _target_: src.optimizer.Qt
  callback:
    _target_: src.util.Callback
    gap: 0.001
    time_limit: 3600
  relax: false
  silent: false
user_groups:
  name: p0
  groups:
  - name: best-effort
    weight: 1
    slo_lower:
      bad: 1
      good: 0
    slo_upper:
      bad: 0
      good: 1
```
