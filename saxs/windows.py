"""The q windows used in training.

Two main windows each receive about 30 % of the samples: [0.05, 0.30] 1/A, used
for the measured curves, and [0.01, 0.30] 1/A, used for the simulated studies. The
rest of each batch cycles through other lower edges. The window edges are given to
the model as context, so one model handles all of them.

Window 0 is the measured-curve window, window 1 the simulation window, and the
remaining windows are the additional lower edges.
"""
from cfm.ingest import WindowedIngest

MAIN_WINDOWS = ((0.05, 0.30), (0.01, 0.30))
P_MAIN = (0.3, 0.3)
Q_HI = 0.30
QLO_SET = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)


class TrainingWindows(WindowedIngest):
    def __init__(self, q_pixel, main_windows=MAIN_WINDOWS, p_main=P_MAIN,
                 qlo_set=QLO_SET, q_hi=Q_HI, **kw):
        super().__init__(q_pixel, real_window=main_windows[0], qlo_set=(), qhi_set=(),
                         jitter=True, p_real=p_main[0], **kw)
        for w in main_windows[1:]:
            self.windows.append(self._build(*w))
        self.n_main = len(main_windows)
        for lo in qlo_set:
            if (lo, q_hi) not in main_windows:
                self.windows.append(self._build(lo, q_hi))
        self.p_main = tuple(p_main)
        assert len(self.windows) > self.n_main, "need at least one jitter window"

    def draw_window_idx(self, rng):
        u, acc = rng.random(), 0.0
        for i, p in enumerate(self.p_main):
            acc += p
            if u < acc:
                return i
        return int(rng.integers(self.n_main, len(self.windows)))
