import tensorflow as tf
import shutil
import os
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
