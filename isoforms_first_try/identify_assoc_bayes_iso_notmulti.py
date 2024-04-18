from move.tasks import identify_associations
from move.data import io

config = io.read_config("isoforms", "isoforms__id_assoc_bayes")
identify_associations(config)

