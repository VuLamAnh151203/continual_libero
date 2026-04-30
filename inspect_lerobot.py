import inspect
try:
    from lerobot.policies.factory import make_policy
    print("Signature of make_policy:")
    print(inspect.signature(make_policy))
except ImportError as e:
    print("ImportError:", e)
