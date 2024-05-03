config = {
    'bootstrap.servers': 'broker:9092',
    'group.id': 'kafka-multi-video4',
    'enable.auto.commit': False,
    'default.topic.config': {'auto.offset.reset': 'earliest'}
}