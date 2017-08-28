def timer(f):
    import time

    def wrapped_func(*args, **kwargs):
        start = time.time()
        val = f(*args, **kwargs)
        total = time.time() - start

        return val, total

    return wrapped_func