import inspect
try:
    from lerobot.scripts.eval import get_policy, load_policy
except ImportError:
    pass
try:
    from lerobot.policies.factory import make_policy
    # Print what make_policy does
    print(inspect.getsource(make_policy))
except Exception as e:
    print(e)
