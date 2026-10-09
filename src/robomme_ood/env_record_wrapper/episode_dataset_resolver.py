# Borrowed: alias of the same-named robomme module; logic follows upstream; manifest: ../UPSTREAM.json
import importlib, sys
sys.modules[__name__] = importlib.import_module("robomme.env_record_wrapper.episode_dataset_resolver")
