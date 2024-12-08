import json

from rec.rec_unit import REC
from utility import configuration

if __name__ == "__main__":
    rec_setup = json.loads(open(configuration.config.get("rec", "setup_file")).read())
    rec = REC.build(rec_setup)
    rec.evaluate()
    rec.write_out()
