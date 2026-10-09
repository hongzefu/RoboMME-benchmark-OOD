# Borrowed: alias of the same-named robomme module; logic follows upstream; manifest: ../../UPSTREAM.json
import importlib, sys
sys.modules[__name__] = importlib.import_module("robomme.robomme_env.utils.obschange")
