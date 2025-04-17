import numpy as np

class MRACExosystem:
    """
    Exosystem generator for structured disturbance signals or reference inputs,
    designed for LunarLander MRAC control.
    """

    def __init__(self, mode="sin", amplitude=0.1, frequency=0.1, dim=2):
        """
        Args:
            mode (str): Type of exosystem signal. Options: 'sin', 'step', 'ramp', 'const'
            amplitude (float): Amplitude of the signal
            frequency (float): Frequency (only used for 'sin')
            dim (int): Dimension of the exosystem state/output
        """
        self.mode = mode
        self.amplitude = amplitude
        self.frequency = frequency
        self.dim = dim
        self.k = 0  # time step

        # Initialize state and system matrix
        self.reset()

        if mode == "sin":#周期扰动（如引擎振荡）
            # 2D oscillator for sinusoidal motion
            omega = frequency
            self.S = np.array([[np.cos(omega), np.sin(omega)],
                               [-np.sin(omega), np.cos(omega)]])
            if dim != 2:
                raise ValueError("Sine exosystem must have dim=2.")
        elif mode in ["step", "const", "ramp"]:
            # Step跳跃扰动, constant恒定偏差（如偏移初始条件）, or ramp线性趋势扰动（如地形倾斜） exosystem
            self.S = np.eye(dim)
        else:
            raise ValueError(f"Unsupported exosystem mode: {mode}")

    def reset(self):
        self.k = 0
        if self.mode == "sin":
            self.state = self.amplitude * np.array([1.0, 0.0])
        elif self.mode == "step":
            self.state = self.amplitude * np.ones(self.dim)
        elif self.mode == "const":
            self.state = self.amplitude * np.ones(self.dim)
        elif self.mode == "ramp":
            self.state = np.zeros(self.dim)
        else:
            raise ValueError("Invalid mode in reset")

    def step(self):
        self.k += 1

        if self.mode == "sin":
            self.state = self.S @ self.state
        elif self.mode == "step":
            pass  # state remains constant
        elif self.mode == "const":
            pass  # state remains constant
        elif self.mode == "ramp":
            self.state += self.amplitude * np.ones(self.dim)
        return self.get()

    def get(self):
        return self.state.copy()
