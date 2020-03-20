# !/usr/bin/env python3

# This file both starts the celery app and it does also define the tasks. These might
# better be split in the future, if the tasks get to large.

from __future__ import absolute_import, unicode_literals
from celery import Celery, signals
from celery_pool_asyncio import monkey as cpa_monkey

# Starting Celery
cpa_monkey.patch()

app = Celery('pycelery')

app.control.purge()

# Setting configurations of celery.
app.conf.update(
    broker_url = 'pyamqp://localhost:5672//',
    result_backend = 'rpc://',
    task_serializer = 'yaml',
    result_serializer = 'json',
    accept_content = ['yaml', 'json'],
    enable_utc = True,
    result_expires = 3600,
    include = 'pycelery.tasks',
    worker_prefetch_multiplier = 2, # contacts works aslong as multiplier x child < 8
    task_acks_late = True,
    task_default_queue = 'robots',
    max_tasks_per_child = 2, # contacts worked with child = 1
    task_routes = ([
    ('pycelery.tasks.*', {'queue': 'robots'}),
    ('pycelery.tasks.hello', {'queue': 'celery'})],)
)

@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    """This function disables logging."""
    pass
    
app.log.setup()

if __name__ == '__main__':
    app.start()
