# DQN + MRAC LunarLander Control Project

This project implements a hybrid control strategy that combines Deep Q-Networks (DQN) with Model Reference Adaptive Control (MRAC) for the LunarLander environment. It enables training and testing of adaptive controllers that leverage reinforcement learning for reference model selection.

---

## Project Structure

```
.
├── DQN/                   # DQN module (agent, network, training logic)
├── env/                   # Custom Gym environments with MRAC wrappers
├── MRAC/                  # MRAC controller implementations
├── models/                # Directory for saved models
├── videos/                # (Optional) rendered test episodes
├── requirements.txt       # Dependency list
├── README.md              # Project documentation
└── Report.pdf             # Technical report
```

---

## Environment Setup

We recommend using Conda to create a clean Python environment:

```bash
conda create -n rl_mrac python=3.10
conda activate rl_mrac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

Edit the `train_mode` variable in `DQN/dqn_main.py` to switch between training and testing:

```python
# Available modes:
train_mode = 'exosystem'  # Train with MRAC controller
# train_mode = 'test'     # Evaluate saved model
```

Then run the main script:

```bash
python -m DQN.dqn_main
```

---

##  Mode Descriptions

| Parameter     | Value         | Description                              |
|---------------|---------------|------------------------------------------|
| `train_mode`  | `'exosystem'` | Start training with MRAC                 |
| `train_mode`  | `'test'`      | Load and evaluate the saved model        |

---

## Notes

- `Box2D` is required for running the LunarLander environment.
- This setup is tested with Python 3.10 and `numpy==1.23.5`.
- Trained models are saved in the `./models` folder. You can customize the save path in the script.

---

# lunarlander_DQN_MRAC
