from move.tasks import identify_associations_multiprocess
from move.data import io

config = io.read_config("isoforms", "isoforms__id_assoc_bayes")
identify_associations_multiprocess(config)

