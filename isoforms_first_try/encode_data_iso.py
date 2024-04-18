from move.tasks import encode_data
from move.data import io
from move.data import preprocessing

config = io.read_config("isoforms", "encode_data")
encode_data(config.data)

