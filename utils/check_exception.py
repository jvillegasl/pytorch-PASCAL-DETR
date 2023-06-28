def check_exception(func, exception):
    try:
        func()
        return False
    except exception:
        return True
