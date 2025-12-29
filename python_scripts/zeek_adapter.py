import os, json, time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka import KafkaProducer

KAFKA_SERVER = os.environ.get('KAFKA_SERVER', 'kafka:29092')
LOG_TOPIC = os.environ.get('LOG_TOPIC', 'network_logs')
ZEEK_LOG_PATH = os.environ.get('ZEEK_LOG_PATH', '/zeek_logs/current')

class ZeekHandler(FileSystemEventHandler):
    def __init__(self, producer):
        self.producer = producer
        self.file_positions = {}

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.log'):
            self.read_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.log'):
            self.file_positions[event.src_path] = 0
            self.read_file(event.src_path)

    def read_file(self, file_path):
        last_pos = self.file_positions.get(file_path, 0)
        try:
            with open(file_path, 'r') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                self.file_positions[file_path] = f.tell()
                
                for line in new_lines:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    try:
                        data = json.loads(line)
                        self.producer.send(LOG_TOPIC, value=data)
                    except: pass
        except FileNotFoundError: pass

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    event_handler = ZeekHandler(producer)
    observer = Observer()
    observer.schedule(event_handler, path=ZEEK_LOG_PATH, recursive=False)
    observer.start()
    print("Zeek Adapter running...")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: pass
    observer.stop()
    producer.flush()
