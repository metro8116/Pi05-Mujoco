"""Decode recorded RGB videos sequentially for episode replay.

Keep one PyAV decoder open per reader and seek to the episode's starting
timestamp, including when multiple LeRobot v3 episodes share one MP4.
Return RGB frames for strictly increasing timestamps and check their
alignment against the dataset frame rate. The replay script uses this
helper to avoid reopening video files for every displayed frame.
"""
from pathlib import Path


class SequentialVideoReader:
    def __init__(self, path, start_timestamp, fps):
        import av

        self.path = Path(path)
        self.container = av.open(str(self.path))
        try:
            self.stream = self.container.streams.video[0]
            # Two persistent decoder threads avoid repeatedly creating large
            # decoder pools on machines with many CPU cores.
            self.stream.codec_context.thread_count = 2
            self.tolerance = 0.5 / fps + 1e-6
            self.last_query = float('-inf')
            # v3 episodes can share an MP4. Seek once to this episode's start.
            self.container.seek(max(0, int(start_timestamp * av.time_base)), backward=True)
            self.frames = iter(self.container.decode(self.stream))
        except BaseException:
            self.container.close()
            raise

    def read(self, timestamp):
        timestamp = float(timestamp)
        if timestamp <= self.last_query:
            raise ValueError('视频时间戳必须递增；切换或重播 episode 时请重新创建 reader')
        for frame in self.frames:
            if frame.time is None:
                continue
            actual = float(frame.time)
            if actual < timestamp - self.tolerance:
                continue
            if abs(actual - timestamp) > self.tolerance:
                raise ValueError(f'{self.path}: 视频与数据不同步，期望 {timestamp:.6f}s，实际 {actual:.6f}s')
            self.last_query = timestamp
            return frame.to_ndarray(format='rgb24')
        raise EOFError(f'{self.path}: 无法读取 {timestamp:.6f}s 的视频帧')

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
