import numpy as np

class Logger:

    def __init__(self, logs=None, **kwargs):
        """
        A logger used to log information during training.

        Args:
            logs: A python dictionary. The keys are the names of the stats to be logged. The values are expected to be empty lists.
        """
        if logs:
            self.logs = logs
        else:
            self.logs = kwargs
    
    def append(self, new_logs=None, **kwargs):
        """
        Append new values to the logger.

        Args:
            new_logs: A python dictionary. The keys are the names of the stats to be logged.
                The values are python integers or floats.
        """
        new_logs = new_logs if new_logs else kwargs
        for key in new_logs:
            if key not in self.logs:
                raise KeyError("Cannot find key, {}, in the logs.".format(key))
            self.logs[key].append(new_logs[key])


    def save_logs(self, path):
        """Save logs to the given path as a compressed npz file.
        """
        np.savez_compressed(
            path,
            **self.logs
        )