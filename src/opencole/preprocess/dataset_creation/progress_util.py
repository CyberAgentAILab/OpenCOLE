import asyncio
from asyncio import Future, Queue
from typing import AsyncIterator, Union

from loguru import logger
from pinjected import injected
from tqdm import tqdm


@injected
async def a_map_progress__tqdm(
    async_f: callable,
    tasks: Union[AsyncIterator, list],
    desc: str,
    pool_size: int = 16,
    total=None,
):
    if isinstance(tasks, list):
        total = len(tasks)
    bar = tqdm(total=total, desc=desc)
    queue = Queue()
    result_queue = Queue()
    if isinstance(tasks, list):
        src_items = tasks

        async def agen():
            for t in src_items:
                yield t

        tasks = agen()

    producer_status = "not started"

    async def producer():
        nonlocal producer_status
        logger.info(f"starting producer with {tasks}")
        producer_status = "started"
        async for task in tasks:
            fut = Future()
            # logger.info(f"producing:{task}")
            producer_status = "submitting"
            await queue.put((fut, task))
            producer_status = "submitted"
            await result_queue.put(fut)
            producer_status = "result future added"
            await asyncio.sleep(0)
        producer_status = "done"
        for _ in range(pool_size):
            await queue.put((False, None))
        await result_queue.put(None)
        producer_status = "finish signal submitted"
        return "producer done"

    consumer_status = dict()

    async def consumer(idx):
        logger.info("starting consumer")
        consumer_status[idx] = "started"
        while True:
            consumer_status[idx] = "waiting"
            match await queue.get():
                case (Future() as fut, task):
                    # logger.info(f"consuming {task}")
                    try:
                        consumer_status[idx] = "running"
                        res = await async_f(task)
                    except Exception as e:
                        fut.set_exception(e)
                        continue
                    bar.update(1)
                    fut.set_result(res)
                case (False, None):
                    consumer_status[idx] = "done"
                    break
        return "consumer done"

    producer_task = producer()
    consumer_tasks = [consumer(idx) for idx in range(pool_size)]
    producer_task = asyncio.create_task(producer_task)
    consumer_task = asyncio.gather(*[asyncio.create_task(t) for t in consumer_tasks])

    while True:
        get_item = asyncio.create_task(result_queue.get())
        done = await get_item
        if done is None:
            break
        else:
            yield await done
    await producer_task
    await consumer_task


@injected
async def a_map_progress__tqdm_serial(
    async_f: callable, tasks: Union[AsyncIterator, list], desc: str, total=None
):
    if isinstance(tasks, list):
        total = len(tasks)

        async def gen():
            for t in tasks:
                yield t

        task_aiter = gen()
    else:
        task_aiter = tasks
    bar = tqdm(total=total, desc=desc)
    async for task in task_aiter:
        res = await async_f(task)
        bar.update(1)
        yield res
