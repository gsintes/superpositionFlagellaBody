"""Test for analysis angle."""

import numpy as np

import analysis_angle as ana

class TestDetection:
    period = 2
    amplitude = 3
    X = np.linspace(0, 20, 1600)
    Y = amplitude * np.sin(2 * np.pi * X / period)
    ANALYSIS = ana.Analysis((0, 1500), angles=Y, times=X)
    ANALYSIS._clean_data()
    ANALYSIS._detect_extrema()

    def test_period(self):
        """Test the period detection."""
        self.ANALYSIS._get_period()

        assert self.ANALYSIS.period < 1.1 * self.period
        assert self.ANALYSIS.period > 0.9 * self.period

    def test_amplitude(self):
        """Test the period detection."""
        self.ANALYSIS._get_amplitude()

        assert self.ANALYSIS.amplitude < 1.1 * self.amplitude
        assert self.ANALYSIS.amplitude > 0.9 * self.amplitude