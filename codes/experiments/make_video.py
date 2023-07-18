import cv2


class MakeVideo(object):
    def __init__(self, width, height, fps, path):

        self.frames = []
        self.width = width
        self.height = height
        codec = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

        self.video_writer = cv2.VideoWriter(path, codec, fps, (width, height))

    def add_frame(self, frame):

        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame.copy(), dsize=(self.width, self.height))
        self.frames.append(frame)

        if len(self.frames) >= 300:
            self.frame_flush()

    def frame_flush(self):
        for frame in self.frames:
            self.video_writer.write(frame)
        self.frames = []

    def stop_and_release(self):

        if self.frames:
            self.frame_flush()

        self.video_writer.release()