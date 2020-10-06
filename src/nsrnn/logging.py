import datetime
import json

class Logger:

    def log(self, event_type, data=None):
        raise NotImplementedError

class NullLogger(Logger):

    def log(self, event_type, data=None):
        pass

class FileLogger(Logger):

    def __init__(self, file, flush=False, reopen=False):
        super().__init__()
        self.file = file
        self.flush = flush
        self.reopen = reopen

    def log(self, event_type, data=None):
        timestamp = str(get_current_timestamp())
        self.file.write(event_type)
        self.file.write(' ')
        self.file.write(timestamp)
        if data is not None:
            self.file.write(' ')
            json.dump(data, self.file, separators=(',', ':'), sort_keys=True)
        self.file.write('\n')
        if self.reopen:
            self.file.close()
            self.file = open(self.file.name, 'a')
        elif self.flush:
            self.file.flush()

def read_log_file(file):
    return map(parse_log_line, file)

def parse_log_line(line):
    fields = line.split(' ', 2)
    try:
        event_type, timestamp_str = fields
    except ValueError:
        event_type, timestamp_str, data_str = fields
        data = json.loads(data_str)
    else:
        data = None
    timestamp = parse_timestamp(float(timestamp_str))
    return ModelLogEvent(event_type, timestamp, data)

class ModelLogEvent:

    def __init__(self, type, timestamp, data):
        self.type = type
        self.timestamp = timestamp
        self.data = data

def get_current_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).timestamp()

def parse_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
