import functools

def to_numpy(func):
    """
    Decorator that converts input arguments and output of a function to numpy arrays.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def to_numpy(obj):
            if hasattr(obj, 'detach') and callable(getattr(obj, 'detach')):
                detached = obj.detach()
                if hasattr(detached, 'numpy') and callable(getattr(detached, 'numpy')):
                    return detached.numpy()
            return obj

        new_args = [to_numpy(arg) for arg in args]
        new_kwargs = {k: to_numpy(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return wrapper